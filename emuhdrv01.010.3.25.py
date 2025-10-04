#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MIPSEMU 2.6-ULTRA64 — Enhanced Boot + Fast-Path (Optimized)  ✅
- Correct MIPS delay-slot handling (kept)
- Proper ROM header parsing & endianness normalization (kept)
- **Fast-path** memory access (no struct slicing; fewer allocations)
- **0xA400_0000 SP mirror mapping fixed** (CPU can actually fetch from DMEM/IMEM)
- Reduced attribute lookups / tighter hot loops
- Adaptive CPU budget per frame (smoother control; perf_counter_ns)
- Coarser canvas tiles by default (less Tk work; still lively)
- PJ64 Bridge seam (ctypes stub) — optional native core drop-in later

NOTE ON PJ64:
This Python UI cannot directly embed Project64’s C/C++ core without a small DLL bridge
with a stable API. This file includes a minimal CoreBridge + PJ64Bridge stub:
  - ENGINE="auto" tries NativeCore first; falls back to PythonCore if unavailable.
  - To wire PJ64, provide a DLL exporting:
      int  Core_Init(void);
      int  Core_LoadROM(const char* path);
      int  Core_RunForCycles(unsigned cycles);      // or frames; adapt the stub
      int  Core_GetFramebuffer(void* out_rgb24, int capacity, int* out_w, int* out_h);
      void Core_Stop(void);
      void Core_Reset(void);
  - Then set ENGINE="native". Until then, the Python core runs (optimized).
"""

import ctypes
import os
import sys
import time
import threading
import hashlib
import struct
from datetime import datetime
from pathlib import Path

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except Exception:
    NUMPY_AVAILABLE = False

# ---------------------------------------------------------------------------
# Engine selection: "auto" (native if available else python), "python", "native"
# ---------------------------------------------------------------------------
ENGINE = os.environ.get("ULTRA64_ENGINE", "auto").lower()  # "auto" | "python" | "native"

# Draw tiles (coarser → fewer rectangles → faster)
DRAW_TILE = 16  # try 8 on fast machines; 16 is ~300 rects/frame
TARGET_FPS = 60


# ===========================================================================
# ROM HEADER (endianness-aware)
# ===========================================================================
class ROMHeader:
    def __init__(self, data: bytes):
        self.raw_data = data[:0x1000]  # quick inspection window
        self.valid = False
        self.endian = "unknown"
        self.clock_rate = 0
        self.boot_address = 0
        self.release = 0
        self.crc1 = 0
        self.crc2 = 0
        self.name = ""
        self.game_code = ""
        self.region = ""
        self.version = 0
        self.rom_hash = ""
        self.parse()

    @staticmethod
    def _swap_endian_pairs(data: bytes) -> bytes:
        # 32-bit word swap: little <-> big
        out = bytearray(len(data))
        pack_le = struct.pack
        unpack_be = struct.unpack
        for i in range(0, len(data), 4):
            if i + 3 < len(data):
                out[i:i+4] = pack_le('<I', unpack_be('>I', data[i:i+4])[0])
        return bytes(out)

    @staticmethod
    def _swap_bytes(data: bytes) -> bytes:
        # v64 byte swap (pairwise)
        out = bytearray(len(data))
        blen = len(data)
        rng = range(0, blen - (blen % 2), 2)
        for i in rng:
            out[i] = data[i+1]
            out[i+1] = data[i]
        if blen & 1:
            out[-1] = data[-1]
        return bytes(out)

    def parse(self):
        if len(self.raw_data) < 0x40:
            return

        magic_be = struct.unpack('>I', self.raw_data[0:4])[0]
        data = self.raw_data

        # Normalize to a big-endian view for parsing
        if magic_be == 0x80371240:       # z64 (big)
            self.endian = 'big'
        elif magic_be == 0x40123780:     # n64 (little)
            self.endian = 'little'
            data = self._swap_endian_pairs(data)
        elif magic_be == 0x37804012:     # v64 (byte-swapped)
            self.endian = 'byteswap'
            data = self._swap_bytes(data)
        else:
            return

        # Parse from big-endian view
        u32 = lambda a, b: struct.unpack('>I', data[a:b])[0]
        self.clock_rate   = u32(0x04, 0x08)
        self.boot_address = u32(0x08, 0x0C)
        self.release      = u32(0x0C, 0x10)
        self.crc1         = u32(0x10, 0x14)
        self.crc2         = u32(0x14, 0x18)
        self.name         = data[0x20:0x34].decode('ascii', errors='ignore').strip('\x00')

        # Correct field widths (cart header spec)
        self.game_code    = data[0x3B:0x3D].decode('ascii', errors='ignore')
        self.region       = chr(data[0x3E])
        self.version      = data[0x3F]

        self.rom_hash     = hashlib.md5(self.raw_data).hexdigest()
        self.valid        = True

    # Full ROM normalization
    def fix_rom_endianness(self, rom: bytes) -> bytes:
        if self.endian == 'little':
            return self._swap_endian_pairs(rom)
        if self.endian == 'byteswap':
            return self._swap_bytes(rom)
        return rom


# ===========================================================================
# COP0 (minimal, enough for boot)
# ===========================================================================
class COP0:
    def __init__(self):
        self.registers = [0] * 32
        self.registers[12] = 0x34000000  # Status: kernel, IE=COP usable
        self.registers[15] = 0x00000B00  # PRId VR4300
        self.registers[16] = 0x7006E463  # Config

    def read_register(self, reg: int) -> int:
        if reg == 1:  # Random
            # cheap-ish jitter; stable enough for a stub
            return int(time.perf_counter_ns() // 1_000_000) % 32
        return self.registers[reg] if 0 <= reg < 32 else 0

    def write_register(self, reg: int, value: int):
        if 0 <= reg < 32 and reg not in (0, 1, 15):
            self.registers[reg] = value & 0xFFFFFFFF


# ===========================================================================
# Memory map (fast-path + fixed SP mirror at 0xA400_0000)
# ===========================================================================
class Memory:
    __slots__ = (
        "rdram", "rom", "rom_size", "sp_dmem", "sp_imem", "pif_rom", "pif_ram", "vi_regs"
    )

    def __init__(self):
        self.rdram   = bytearray(8 * 1024 * 1024)  # include expansion; easy
        self.rom     = None
        self.rom_size = 0
        self.sp_dmem = bytearray(4096)
        self.sp_imem = bytearray(4096)
        self.pif_rom = bytearray(2048)
        self.pif_ram = bytearray(64)
        self.vi_regs = [0] * 32

    def load_rom(self, rom_data: bytes):
        self.rom = bytearray(rom_data)
        self.rom_size = len(self.rom)

    # --- helpers (big-endian) ---
    @staticmethod
    def _be16_from(buf, off):
        return (buf[off] << 8) | buf[off + 1]

    @staticmethod
    def _be32_from(buf, off):
        return (buf[off] << 24) | (buf[off + 1] << 16) | (buf[off + 2] << 8) | buf[off + 3]

    @staticmethod
    def _be16_to(buf, off, value):
        buf[off]     = (value >> 8) & 0xFF
        buf[off + 1] = value & 0xFF

    @staticmethod
    def _be32_to(buf, off, value):
        buf[off]     = (value >> 24) & 0xFF
        buf[off + 1] = (value >> 16) & 0xFF
        buf[off + 2] = (value >> 8) & 0xFF
        buf[off + 3] = value & 0xFF

    @staticmethod
    def _mask32(addr: int) -> int:
        return addr & 0xFFFFFFFF

    # --- fast-path reads ---
    def read_byte(self, addr: int) -> int:
        a = self._mask32(addr)

        # RDRAM mirrors
        if a < 0x00800000:
            return self.rdram[a]
        if 0x80000000 <= a < 0x80800000:
            return self.rdram[a & 0x007FFFFF]
        if 0xA0000000 <= a < 0xA0800000:
            return self.rdram[a & 0x007FFFFF]

        # SP DMEM/IMEM — both 0x0400_0000 and 0xA400_0000 mirrors
        if (0x04000000 <= a < 0x04002000) or (0xA4000000 <= a < 0xA4002000):
            lo = a & 0x1FFF
            if lo < 0x1000:
                return self.sp_dmem[lo & 0xFFF]
            else:
                return self.sp_imem[lo & 0xFFF]

        # Cartridge
        if 0x10000000 <= a < 0x1FC00000:
            off = a & 0x0FFFFFFF
            return self.rom[off] if self.rom and off < self.rom_size else 0
        if 0xB0000000 <= a < 0xBFC00000:
            off = a & 0x0FFFFFFF
            return self.rom[off] if self.rom and off < self.rom_size else 0

        # PIF
        if 0x1FC00000 <= a < 0x1FC007C0:
            return self.pif_rom[a & 0x7FF]
        if 0x1FC007C0 <= a < 0x1FC00800:
            return self.pif_ram[a & 0x3F]

        # VI (very coarse)
        if 0x04400000 <= a < 0x04400080:
            idx = (a - 0x04400000) >> 2
            return (self.vi_regs[idx & 0x1F] & 0xFF)

        return 0

    def read_half(self, addr: int) -> int:
        a = self._mask32(addr) & 0xFFFFFFFE

        if a < 0x00800000:
            if a + 1 < len(self.rdram):
                return self._be16_from(self.rdram, a)
            return 0
        if 0x80000000 <= a < 0x80800000:
            off = a & 0x007FFFFE
            return self._be16_from(self.rdram, off)
        if 0xA0000000 <= a < 0xA0800000:
            off = a & 0x007FFFFE
            return self._be16_from(self.rdram, off)

        if (0x04000000 <= a < 0x04002000) or (0xA4000000 <= a < 0xA4002000):
            lo = a & 0x1FFF
            if lo + 1 < 0x1000:
                return self._be16_from(self.sp_dmem, lo & 0xFFF)
            elif 0x1000 <= lo < 0x2000-1:
                return self._be16_from(self.sp_imem, lo & 0xFFF)
            return 0

        if 0x10000000 <= a < 0x1FC00000 or 0xB0000000 <= a < 0xBFC00000:
            off = a & 0x0FFFFFFE
            return self._be16_from(self.rom, off) if self.rom and off + 1 < self.rom_size else 0

        return 0

    def read_word(self, addr: int) -> int:
        a = self._mask32(addr) & 0xFFFFFFFC

        # RDRAM
        if a < 0x00800000:
            if a + 3 < len(self.rdram):
                return self._be32_from(self.rdram, a)
            return 0
        if 0x80000000 <= a < 0x80800000:
            off = a & 0x007FFFFC
            return self._be32_from(self.rdram, off)
        if 0xA0000000 <= a < 0xA0800000:
            off = a & 0x007FFFFC
            return self._be32_from(self.rdram, off)

        # SP mirrors
        if (0x04000000 <= a < 0x04002000) or (0xA4000000 <= a < 0xA4002000):
            lo = a & 0x1FFF
            if lo + 3 < 0x1000:
                return self._be32_from(self.sp_dmem, lo & 0xFFF)
            elif 0x1000 <= lo <= 0x1FFC:
                return self._be32_from(self.sp_imem, lo & 0xFFF)
            return 0

        # VI
        if 0x04400000 <= a < 0x04400080:
            idx = (a - 0x04400000) >> 2
            return self.vi_regs[idx & 0x1F]

        # Cartridge
        if 0x10000000 <= a < 0x1FC00000:
            off = a & 0x0FFFFFFC
            return self._be32_from(self.rom, off) if self.rom and off + 3 < self.rom_size else 0
        if 0xB0000000 <= a < 0xBFC00000:
            off = a & 0x0FFFFFFC
            return self._be32_from(self.rom, off) if self.rom and off + 3 < self.rom_size else 0

        # PIF ROM is byte/half read mostly; word reads fall back to 0
        return 0

    # --- writes ---
    def write_byte(self, addr: int, value: int):
        a = self._mask32(addr)
        v = value & 0xFF

        if a < 0x00800000:
            self.rdram[a] = v; return
        if 0x80000000 <= a < 0x80800000 or 0xA0000000 <= a < 0xA0800000:
            self.rdram[a & 0x007FFFFF] = v; return

        if (0x04000000 <= a < 0x04002000) or (0xA4000000 <= a < 0xA4002000):
            lo = a & 0x1FFF
            if lo < 0x1000:
                self.sp_dmem[lo & 0xFFF] = v
            else:
                self.sp_imem[lo & 0xFFF] = v
            return

        if 0x1FC007C0 <= a < 0x1FC00800:
            self.pif_ram[a & 0x3F] = v
            return

        if 0x04400000 <= a < 0x04400080:
            # ignore for now (would update VI regs)
            return

    def write_half(self, addr: int, value: int):
        a = self._mask32(addr) & 0xFFFFFFFE
        v = value & 0xFFFF

        if a < 0x00800000:
            if a + 1 < len(self.rdram): self._be16_to(self.rdram, a, v); return
        elif 0x80000000 <= a < 0x80800000 or 0xA0000000 <= a < 0xA0800000:
            off = a & 0x007FFFFE; self._be16_to(self.rdram, off, v); return
        elif (0x04000000 <= a < 0x04002000) or (0xA4000000 <= a < 0xA4002000):
            lo = a & 0x1FFF
            if lo + 1 < 0x1000:
                self._be16_to(self.sp_dmem, lo & 0xFFF, v); return
            if 0x1000 <= lo < 0x2000 - 1:
                self._be16_to(self.sp_imem, lo & 0xFFF, v); return
        elif 0x04400000 <= a < 0x04400080:
            return  # ignore
        # no cart writes in this stub

    def write_word(self, addr: int, value: int):
        a = self._mask32(addr) & 0xFFFFFFFC
        v = value & 0xFFFFFFFF

        if a < 0x00800000:
            if a + 3 < len(self.rdram): self._be32_to(self.rdram, a, v); return
        elif 0x80000000 <= a < 0x80800000 or 0xA0000000 <= a < 0xA0800000:
            off = a & 0x007FFFFC; self._be32_to(self.rdram, off, v); return
        elif (0x04000000 <= a < 0x04002000) or (0xA4000000 <= a < 0xA4002000):
            lo = a & 0x1FFF
            if lo + 3 < 0x1000:
                self._be32_to(self.sp_dmem, lo & 0xFFF, v); return
            if 0x1000 <= lo <= 0x1FFC:
                self._be32_to(self.sp_imem, lo & 0xFFF, v); return
        elif 0x04400000 <= a < 0x04400080:
            idx = (a - 0x04400000) >> 2
            self.vi_regs[idx & 0x1F] = v; return
        # cart write ignored


# ===========================================================================
# PIF (very light boot shim)
# ===========================================================================
class PIF:
    def __init__(self, memory: Memory):
        self.memory = memory

    def simulate_boot(self, rom_header: ROMHeader) -> bool:
        if not rom_header or not rom_header.valid:
            return False
        # Mock IPL3 presence by copying the first 0x1000 bytes into DMEM
        chunk = rom_header.raw_data[:0x1000]
        self.memory.sp_dmem[:len(chunk)] = chunk
        # Minimal PIF RAM init
        self.memory.pif_ram[0x24:0x28] = b'\x00\x00\x00\x3F'
        return True


# ===========================================================================
# RDP stub (simple framebuffer we can draw)
# ===========================================================================
class RDP:
    def __init__(self):
        if NUMPY_AVAILABLE:
            self.framebuffer = np.zeros((240, 320, 3), dtype=np.uint8)
        else:
            self.framebuffer = [[(0, 0, 0) for _ in range(320)] for _ in range(240)]

    def clear_framebuffer(self, color=(0, 0, 0)):
        if NUMPY_AVAILABLE:
            self.framebuffer[:] = color
        else:
            c = tuple(color)
            fb = self.framebuffer
            for y in range(240):
                row = fb[y]
                for x in range(320):
                    row[x] = c


# ===========================================================================
# MIPS R4300i (optimized step; delay slot kept)
# ===========================================================================
class MIPSCPU:
    __slots__ = (
        "memory", "pc", "next_pc", "registers", "hi", "lo", "cop0",
        "running", "instructions_executed", "cycles",
        "branch_pending", "branch_target", "in_delay_slot", "ll_bit"
    )

    def __init__(self, memory: Memory):
        self.memory = memory
        self.pc = 0xA4000040
        self.next_pc = self.pc + 4
        self.registers = [0] * 32
        self.hi = 0
        self.lo = 0
        self.cop0 = COP0()

        self.running = False
        self.instructions_executed = 0
        self.cycles = 0

        self.branch_pending = False
        self.branch_target = 0
        self.in_delay_slot = False

        self.ll_bit = 0  # not used

    # ---------- helpers ----------
    @staticmethod
    def _signed32(v: int) -> int:
        return v - 0x100000000 if (v & 0x80000000) else v

    @staticmethod
    def _sx8(v: int) -> int:
        return (v | ~0xFF) if (v & 0x80) else (v & 0xFF)

    @staticmethod
    def _sx16(v: int) -> int:
        return (v | ~0xFFFF) if (v & 0x8000) else (v & 0xFFFF)

    def reset(self):
        self.pc = 0xA4000040
        self.next_pc = self.pc + 4
        self.registers = [0] * 32
        self.hi = 0
        self.lo = 0
        self.cop0 = COP0()
        self.instructions_executed = 0
        self.cycles = 0
        self.branch_pending = False
        self.in_delay_slot = False
        self.running = False

    def boot_setup(self, boot_address: int):
        # On real HW we start in PIF; then jump to IPL3/boot
        self.pc = boot_address or 0xA4000040
        self.next_pc = (self.pc + 4) & 0xFFFFFFFF
        self.registers[29] = 0xA4001FF0  # SP
        self.registers[31] = 0xA4001550  # RA
        self.running = True

    # ---------- main step ----------
    def step(self):
        if not self.running:
            return

        mem = self.memory
        read_word = mem.read_word
        read_byte = mem.read_byte
        read_half = mem.read_half
        write_word = mem.write_word
        write_byte = mem.write_byte
        write_half = mem.write_half

        regs = self.registers
        s32 = self._signed32
        sx16 = self._sx16
        sx8 = self._sx8

        try:
            instr = read_word(self.pc)
            opcode = (instr >> 26) & 0x3F
            rs = (instr >> 21) & 0x1F
            rt = (instr >> 16) & 0x1F
            rd = (instr >> 11) & 0x1F
            shamt = (instr >> 6) & 0x1F
            funct = instr & 0x3F
            imm = instr & 0xFFFF
            target = instr & 0x03FFFFFF

            self.next_pc = (self.pc + 4) & 0xFFFFFFFF
            do_branch = False
            new_target = 0

            if opcode == 0x00:
                # SPECIAL
                if funct == 0x00:       # SLL
                    if instr != 0:      # NOP guard
                        regs[rd] = (regs[rt] << shamt) & 0xFFFFFFFF
                elif funct == 0x02:     # SRL
                    regs[rd] = (regs[rt] >> shamt) & 0xFFFFFFFF
                elif funct == 0x03:     # SRA
                    regs[rd] = (s32(regs[rt]) >> shamt) & 0xFFFFFFFF
                elif funct == 0x04:     # SLLV
                    regs[rd] = (regs[rt] << (regs[rs] & 0x1F)) & 0xFFFFFFFF
                elif funct == 0x06:     # SRLV
                    regs[rd] = (regs[rt] >> (regs[rs] & 0x1F)) & 0xFFFFFFFF
                elif funct == 0x07:     # SRAV
                    regs[rd] = (s32(regs[rt]) >> (regs[rs] & 0x1F)) & 0xFFFFFFFF
                elif funct == 0x08:     # JR
                    new_target = regs[rs] & 0xFFFFFFFF
                    do_branch = True
                elif funct == 0x09:     # JALR
                    regs[rd] = (self.pc + 8) & 0xFFFFFFFF
                    new_target = regs[rs] & 0xFFFFFFFF
                    do_branch = True
                elif funct == 0x10:     # MFHI
                    regs[rd] = self.hi
                elif funct == 0x11:     # MTHI
                    self.hi = regs[rs]
                elif funct == 0x12:     # MFLO
                    regs[rd] = self.lo
                elif funct == 0x13:     # MTLO
                    self.lo = regs[rs]
                elif funct == 0x18:     # MULT
                    result = s32(regs[rs]) * s32(regs[rt])
                    self.lo = result & 0xFFFFFFFF
                    self.hi = (result >> 32) & 0xFFFFFFFF
                elif funct == 0x19:     # MULTU
                    result = (regs[rs] & 0xFFFFFFFF) * (regs[rt] & 0xFFFFFFFF)
                    self.lo = result & 0xFFFFFFFF
                    self.hi = (result >> 32) & 0xFFFFFFFF
                elif funct == 0x1A:     # DIV
                    if regs[rt] != 0:
                        self.lo = (s32(regs[rs]) // s32(regs[rt])) & 0xFFFFFFFF
                        self.hi = (s32(regs[rs]) %  s32(regs[rt])) & 0xFFFFFFFF
                elif funct == 0x1B:     # DIVU
                    if regs[rt] != 0:
                        self.lo = (regs[rs] // regs[rt]) & 0xFFFFFFFF
                        self.hi = (regs[rs] %  regs[rt]) & 0xFFFFFFFF
                elif funct == 0x20 or funct == 0x21:  # ADD/ADDU
                    regs[rd] = (regs[rs] + regs[rt]) & 0xFFFFFFFF
                elif funct == 0x22 or funct == 0x23:  # SUB/SUBU
                    regs[rd] = (regs[rs] - regs[rt]) & 0xFFFFFFFF
                elif funct == 0x24:     # AND
                    regs[rd] = regs[rs] & regs[rt]
                elif funct == 0x25:     # OR
                    regs[rd] = regs[rs] | regs[rt]
                elif funct == 0x26:     # XOR
                    regs[rd] = regs[rs] ^ regs[rt]
                elif funct == 0x27:     # NOR
                    regs[rd] = (~(regs[rs] | regs[rt])) & 0xFFFFFFFF
                elif funct == 0x2A:     # SLT
                    regs[rd] = 1 if s32(regs[rs]) < s32(regs[rt]) else 0
                elif funct == 0x2B:     # SLTU
                    regs[rd] = 1 if (regs[rs] & 0xFFFFFFFF) < (regs[rt] & 0xFFFFFFFF) else 0
                # else: NOP/UNIMPL

            elif opcode == 0x01:  # REGIMM
                offset = (sx16(imm) << 2) & 0xFFFFFFFF
                srs = s32(regs[rs])
                if rt == 0x00 and srs < 0:          # BLTZ
                    new_target = (self.next_pc + offset) & 0xFFFFFFFF; do_branch = True
                elif rt == 0x01 and srs >= 0:       # BGEZ
                    new_target = (self.next_pc + offset) & 0xFFFFFFFF; do_branch = True
                elif rt == 0x10 and srs < 0:        # BLTZAL
                    regs[31] = (self.pc + 8) & 0xFFFFFFFF
                    new_target = (self.next_pc + offset) & 0xFFFFFFFF; do_branch = True
                elif rt == 0x11 and srs >= 0:       # BGEZAL
                    regs[31] = (self.pc + 8) & 0xFFFFFFFF
                    new_target = (self.next_pc + offset) & 0xFFFFFFFF; do_branch = True

            elif opcode == 0x02:  # J
                new_target = ((self.pc & 0xF0000000) | (target << 2)) & 0xFFFFFFFF; do_branch = True
            elif opcode == 0x03:  # JAL
                regs[31] = (self.pc + 8) & 0xFFFFFFFF
                new_target = ((self.pc & 0xF0000000) | (target << 2)) & 0xFFFFFFFF; do_branch = True

            elif opcode == 0x04:  # BEQ
                if regs[rs] == regs[rt]:
                    new_target = (self.next_pc + ((sx16(imm) << 2) & 0xFFFFFFFF)) & 0xFFFFFFFF; do_branch = True
            elif opcode == 0x05:  # BNE
                if regs[rs] != regs[rt]:
                    new_target = (self.next_pc + ((sx16(imm) << 2) & 0xFFFFFFFF)) & 0xFFFFFFFF; do_branch = True
            elif opcode == 0x06:  # BLEZ
                if s32(regs[rs]) <= 0:
                    new_target = (self.next_pc + ((sx16(imm) << 2) & 0xFFFFFFFF)) & 0xFFFFFFFF; do_branch = True
            elif opcode == 0x07:  # BGTZ
                if s32(regs[rs]) > 0:
                    new_target = (self.next_pc + ((sx16(imm) << 2) & 0xFFFFFFFF)) & 0xFFFFFFFF; do_branch = True

            elif opcode == 0x08 or opcode == 0x09:   # ADDI/ADDIU
                regs[rt] = (regs[rs] + sx16(imm)) & 0xFFFFFFFF
            elif opcode == 0x0A:                     # SLTI
                regs[rt] = 1 if s32(regs[rs]) < sx16(imm) else 0
            elif opcode == 0x0B:                     # SLTIU
                regs[rt] = 1 if (regs[rs] & 0xFFFFFFFF) < (imm & 0xFFFF) else 0
            elif opcode == 0x0C:                     # ANDI
                regs[rt] = regs[rs] & (imm & 0xFFFF)
            elif opcode == 0x0D:                     # ORI
                regs[rt] = (regs[rs] | (imm & 0xFFFF)) & 0xFFFFFFFF
            elif opcode == 0x0E:                     # XORI
                regs[rt] = (regs[rs] ^ (imm & 0xFFFF)) & 0xFFFFFFFF
            elif opcode == 0x0F:                     # LUI
                regs[rt] = (imm << 16) & 0xFFFFFFFF

            elif opcode == 0x10:  # COP0
                cop_op = (instr >> 21) & 0x1F
                if cop_op == 0x00:                   # MFC0
                    regs[rt] = self.cop0.read_register(rd)
                elif cop_op == 0x04:                 # MTC0
                    self.cop0.write_register(rd, regs[rt])

            elif opcode == 0x20:  # LB
                addr = (regs[rs] + sx16(imm)) & 0xFFFFFFFF
                regs[rt] = sx8(read_byte(addr))
            elif opcode == 0x21:  # LH
                addr = (regs[rs] + sx16(imm)) & 0xFFFFFFFF
                regs[rt] = sx16(read_half(addr))
            elif opcode == 0x23:  # LW
                addr = (regs[rs] + sx16(imm)) & 0xFFFFFFFF
                regs[rt] = read_word(addr)
            elif opcode == 0x24:  # LBU
                addr = (regs[rs] + sx16(imm)) & 0xFFFFFFFF
                regs[rt] = read_byte(addr) & 0xFF
            elif opcode == 0x25:  # LHU
                addr = (regs[rs] + sx16(imm)) & 0xFFFFFFFF
                regs[rt] = read_half(addr) & 0xFFFF

            elif opcode == 0x28:  # SB
                addr = (regs[rs] + sx16(imm)) & 0xFFFFFFFF
                write_byte(addr, regs[rt] & 0xFF)
            elif opcode == 0x29:  # SH
                addr = (regs[rs] + sx16(imm)) & 0xFFFFFFFF
                write_half(addr, regs[rt] & 0xFFFF)
            elif opcode == 0x2B:  # SW
                addr = (regs[rs] + sx16(imm)) & 0xFFFFFFFF
                write_word(addr, regs[rt])

            # -------- program counter transition w/ delay slot --------
            regs[0] = 0  # $zero enforced

            if self.in_delay_slot:
                # we just executed the delay instruction; now jump
                self.pc = self.branch_target
                self.in_delay_slot = False
                self.branch_pending = False
            else:
                if do_branch:
                    self.branch_pending = True
                    self.branch_target = new_target
                    self.in_delay_slot = True
                self.pc = self.next_pc

            self.instructions_executed += 1
            self.cycles += 1

        except Exception as e:
            print(f"CPU Error at PC={hex(self.pc)}: {e}")
            self.running = False


# ===========================================================================
# Core bridge(s)
# ===========================================================================
class BaseCore:
    def load_rom(self, filepath: str):
        raise NotImplementedError

    def start(self):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def run_for_instructions(self, n: int) -> bool:
        raise NotImplementedError

    def get_regs_snapshot(self):
        raise NotImplementedError

    def get_framebuffer(self):
        raise NotImplementedError

    @property
    def rom_header(self) -> ROMHeader:
        raise NotImplementedError


class PythonCore(BaseCore):
    """Your original emulator core, optimized."""
    def __init__(self):
        self.memory = Memory()
        self.cpu = MIPSCPU(self.memory)
        self.pif = PIF(self.memory)
        self.rdp = RDP()
        self._rom_header = None

    @property
    def rom_header(self) -> ROMHeader:
        return self._rom_header

    def load_rom(self, filepath: str):
        with open(filepath, "rb") as f:
            rom = f.read()

        hdr = ROMHeader(rom)
        if not hdr.valid:
            raise ValueError("Invalid N64 ROM file.")

        rom_fixed = hdr.fix_rom_endianness(rom)
        self.memory.load_rom(rom_fixed)
        self._rom_header = hdr
        return hdr

    def start(self):
        if self.pif.simulate_boot(self._rom_header):
            pass
        self.cpu.boot_setup(self._rom_header.boot_address)

    def stop(self):
        self.cpu.running = False

    def reset(self):
        self.cpu.reset()

    def run_for_instructions(self, n: int) -> bool:
        step = self.cpu.step
        for _ in range(n):
            if not self.cpu.running:
                return False
            step()
        return self.cpu.running

    def get_regs_snapshot(self):
        c = self.cpu
        return {
            "pc": c.pc,
            "sp": c.registers[29],
            "ra": c.registers[31],
            "instr": c.instructions_executed
        }

    def get_framebuffer(self):
        return self.rdp.framebuffer


class NativeCore(BaseCore):
    """
    PJ64Bridge stub: will try to load a native core via ctypes (DLL/.so).
    You must supply a DLL that matches the functions noted at the top.
    Until then, this raises at init or returns fallback.
    """
    def __init__(self):
        self._lib = None
        self._hdr = None
        self._width = 320
        self._height = 240
        self._fb = None  # numpy array or list-based fallback

        # Try to find a plausible library name
        candidates = [
            os.environ.get("ULTRA64_NATIVE_DLL"),
            "pj64s.dll", "pj64s.so", "pj64s.dylib",
            "mupen64plus.dll", "mupen64plus.so", "n64core.dll"
        ]
        candidates = [c for c in candidates if c]
        last_err = None
        for name in candidates:
            try:
                self._lib = ctypes.CDLL(name)
                break
            except Exception as e:
                last_err = e
                self._lib = None
        if not self._lib:
            raise RuntimeError(f"No native core found ({last_err})")

        # TODO: set argtypes/restype here once your bridge DLL is ready
        # self._lib.Core_Init.restype = ctypes.c_int
        # ...

        # Simple local framebuffer for UI if native core cannot supply one
        if NUMPY_AVAILABLE:
            self._fb = np.zeros((240, 320, 3), dtype=np.uint8)
        else:
            self._fb = [[(0, 0, 0) for _ in range(320)] for _ in range(240)]

    @property
    def rom_header(self) -> ROMHeader:
        return self._hdr

    def load_rom(self, filepath: str):
        # We still parse the header here for the UI/log, even if native core runs it.
        with open(filepath, "rb") as f:
            rom = f.read()
        hdr = ROMHeader(rom)
        if not hdr.valid:
            raise ValueError("Invalid N64 ROM file (header).")
        self._hdr = hdr

        # Call into native core (replace with your exported function)
        # rc = self._lib.Core_LoadROM(filepath.encode('utf-8'))
        # if rc != 0: raise RuntimeError("Native core failed to load ROM")

        return hdr

    def start(self):
        # rc = self._lib.Core_Init()
        # if rc != 0: raise RuntimeError("Native core init failed")
        pass

    def stop(self):
        # self._lib.Core_Stop()
        pass

    def reset(self):
        # self._lib.Core_Reset()
        pass

    def run_for_instructions(self, n: int) -> bool:
        # If your native API is "cycles" or "frames", adapt this call.
        # self._lib.Core_RunForCycles(ctypes.c_uint(n))
        return True

    def get_regs_snapshot(self):
        # Would query native core; placeholder values for UI
        return {"pc": 0, "sp": 0, "ra": 0, "instr": 0}

    def get_framebuffer(self):
        # If your core can give us a framebuffer, copy it into self._fb.
        # self._lib.Core_GetFramebuffer(...)
        return self._fb


# ===========================================================================
# Emulator shell (UI + loops)
# ===========================================================================
class N64Emulator:
    def __init__(self, root, engine_mode: str = ENGINE):
        self.root = root
        self.root.title("MIPSEMU 2.6 — Enhanced Boot + Fast-Path")
        self.root.geometry("1024x768")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Core selection
        self.core = None
        if engine_mode == "python":
            self.core = PythonCore()
        elif engine_mode == "native":
            try:
                self.core = NativeCore()
            except Exception as e:
                messagebox.showwarning("Native core unavailable", f"{e}\nFalling back to Python core.")
                self.core = PythonCore()
        else:  # auto
            try:
                self.core = NativeCore()
            except Exception:
                self.core = PythonCore()

        self.emulation_running = False
        self.boot_status = 'idle'
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = time.time()
        self.frames_this_second = 0

        # dynamic CPU budget to target ~60 fps (instructions per frame)
        self.instructions_per_frame = 20000
        self.min_ipf = 2000
        self.max_ipf = 200000

        self._build_ui()

    # ---------- UI ----------
    def _build_ui(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open ROM", command=self.open_rom)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_close)

        emu_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Emulation", menu=emu_menu)
        emu_menu.add_command(label="Start", command=self.start_emulation, accelerator="F5")
        emu_menu.add_command(label="Stop", command=self.stop_emulation, accelerator="F6")
        emu_menu.add_command(label="Reset", command=self.reset_emulation, accelerator="F7")

        toolbar = tk.Frame(self.root, bg="#1e1e1e")
        toolbar.pack(side=tk.TOP, fill=tk.X)
        style_btn = {"bg": "#3c3c3c", "fg": "white", "relief": tk.FLAT, "padx": 10, "pady": 5}
        tk.Button(toolbar, text="Open ROM", command=self.open_rom, **style_btn).pack(side=tk.LEFT, padx=2)
        tk.Button(toolbar, text="Start", command=self.start_emulation, **style_btn).pack(side=tk.LEFT, padx=2)
        tk.Button(toolbar, text="Stop", command=self.stop_emulation, **style_btn).pack(side=tk.LEFT, padx=2)

        self.canvas = tk.Canvas(self.root, bg="#000000", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        log_frame = tk.Frame(self.root, bg="#1e1e1e", height=100)
        self.log_text = scrolledtext.ScrolledText(
            log_frame, bg="#0a0a0a", fg="#00ff00", font=("Consolas", 9), height=8
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        log_frame.pack(side=tk.BOTTOM, fill=tk.X)

        status = tk.Frame(self.root, bg="#1e1e1e")
        status.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_label = tk.Label(status, text="Ready", bg="#1e1e1e", fg="white", anchor=tk.W)
        self.status_label.pack(side=tk.LEFT, padx=10)
        self.fps_label = tk.Label(status, text="FPS: 0", bg="#1e1e1e", fg="#00ff00")
        self.fps_label.pack(side=tk.RIGHT, padx=10)

        # shortcuts
        self.root.bind("<F5>", lambda e: self.start_emulation())
        self.root.bind("<F6>", lambda e: self.stop_emulation())
        self.root.bind("<F7>", lambda e: self.reset_emulation())

        self._log("MIPSEMU 2.6 — Optimized build ready.")
        self._log("Fixes: SP mirror @ 0xA4000000; fast memory; smoother 60 FPS pacing; PJ64 bridge seam.")

    def _log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{ts}] {msg}\n")
        self.log_text.see(tk.END)

    # ---------- ROM ----------
    def open_rom(self):
        filename = filedialog.askopenfilename(
            title="Select N64 ROM",
            filetypes=[("N64 ROMs", "*.z64 *.n64 *.v64"), ("All Files", "*.*")]
        )
        if filename:
            self.load_rom(filename)

    def load_rom(self, filepath: str):
        try:
            name = Path(filepath).name
            self._log(f"Loading: {name}")
            hdr = self.core.load_rom(filepath)
            self._log(f"ROM: {hdr.name}")
            self._log(f"Game Code: {hdr.game_code} | Region: {hdr.region} | Ver: {hdr.version}")
            self._log(f"Format: {hdr.endian}-endian (normalized in-core if needed)")
            self._log(f"Boot Address: {hex(hdr.boot_address)}")
            # We can only report normalized size when PythonCore owns the ROM bytes
            if isinstance(self.core, PythonCore):
                self._log(f"Size: {self.core.memory.rom_size // (1024*1024)} MB")
            self.root.title(f"MIPSEMU 2.6 — {hdr.name}")
            self.status_label.config(text=f"Loaded: {name}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load ROM: {e}")
            self._log(f"ERROR: {e}")

    # ---------- Emulation control ----------
    def start_emulation(self):
        try:
            if not self.core.rom_header:
                messagebox.showwarning("No ROM", "Please load a ROM first.")
                return

            self._log("=" * 64)
            self._log("STARTING BOOT SEQUENCE")
            self._log("=" * 64)
            self.boot_status = 'booting'

            self.core.start()  # PIF shim + CPU PC set (PythonCore) or native init
            regs = self.core.get_regs_snapshot()
            self._log(f"CPU: PC = {hex(regs['pc'])}  SP = {hex(regs['sp'])}")
            self._log("Boot complete — starting execution.")
            self._log("=" * 64)

            self.emulation_running = True
            self.boot_status = 'running'

            self.emu_thread = threading.Thread(target=self._emulation_loop, daemon=True)
            self.emu_thread.start()

            self._render_loop()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to start: {e}")
            self._log(f"ERROR: {e}")

    def stop_emulation(self):
        try:
            self.emulation_running = False
            self.core.stop()
            self.boot_status = 'idle'
            self._log("Emulation stopped.")
        except Exception as e:
            self._log(f"Stop error: {e}")

    def reset_emulation(self):
        try:
            self.stop_emulation()
            self.core.reset()
            self.frame_count = 0
            self.instructions_per_frame = 20000
            self._log("Emulation reset.")
        except Exception as e:
            self._log(f"Reset error: {e}")

    def _on_close(self):
        self.stop_emulation()
        # Give thread a moment to exit
        self.root.after(100, self.root.destroy)

    # ---------- Loops ----------
    def _emulation_loop(self):
        target_dt = 1.0 / TARGET_FPS
        target_ns = int(target_dt * 1_000_000_000)

        while self.emulation_running:
            t0 = time.perf_counter_ns()

            ipf = self.instructions_per_frame
            still_running = self.core.run_for_instructions(ipf)
            if not still_running:
                self._log("Core halted.")
                self.emulation_running = False
                break

            # Adaptive budget to hold ~60 FPS
            frame_ns = time.perf_counter_ns() - t0
            # leniency band
            if frame_ns < int(target_ns * 0.6):
                self.instructions_per_frame = min(self.max_ipf, int(self.instructions_per_frame * 1.15) + 200)
            elif frame_ns > int(target_ns * 1.2):
                self.instructions_per_frame = max(self.min_ipf, int(self.instructions_per_frame * 0.85))

            # Sleep to roughly cap at 60Hz
            sleep_ns = target_ns - frame_ns
            if sleep_ns > 0:
                time.sleep(sleep_ns / 1_000_000_000.0)

    def _render_loop(self):
        if not self.emulation_running:
            return

        try:
            self.canvas.delete("all")

            # bezel
            self.canvas.create_rectangle(0, 0, 1024, 768, fill="#001122", outline="")
            sx, sy = 192, 114
            self.canvas.create_rectangle(sx, sy, sx + 640, sy + 480, fill="#000000", outline="#00ff88", width=2)

            if self.boot_status == 'running':
                fb = self.core.get_framebuffer()
                self._render_framebuffer(sx, sy, fb)

                regs = self.core.get_regs_snapshot()
                self.canvas.create_text(
                    sx + 320, sy + 20,
                    text=f"PC: {hex(regs['pc'])} | Instr: {regs['instr']:,}",
                    font=("Consolas", 10), fill="#00ff00"
                )
                self.canvas.create_text(
                    sx + 320, sy + 40,
                    text=f"SP: {hex(regs['sp'])} | RA: {hex(regs['ra'])}",
                    font=("Consolas", 10), fill="#00ff00"
                )
            elif self.boot_status == 'booting':
                self.canvas.create_text(
                    sx + 320, sy + 240,
                    text="NINTENDO 64",
                    font=("Arial", 48, "bold"), fill="#ff0000"
                )

            # FPS counter
            self.frames_this_second += 1
            now = time.time()
            if now - self.last_fps_time >= 1.0:
                self.fps = self.frames_this_second
                self.fps_label.config(text=f"FPS: {self.fps}")
                self.frames_this_second = 0
                self.last_fps_time = now

            self.frame_count += 1
            # simple animated clear so we see changes even without a real RDP
            if isinstance(self.core, PythonCore) and NUMPY_AVAILABLE and (self.frame_count % 30 == 0):
                c = (
                    (self.frame_count % 256) // 4,
                    ((self.frame_count * 2) % 256) // 4,
                    ((self.frame_count * 3) % 256) // 4
                )
                self.core.rdp.clear_framebuffer(c)

            self.root.after(int(1000 / TARGET_FPS), self._render_loop)

        except Exception as e:
            self._log(f"Render error: {e}")

    def _render_framebuffer(self, x: int, y: int, fb):
        # Coarse tiles (fewer canvas ops). Tile size via DRAW_TILE.
        scale = 2
        ts = DRAW_TILE
        create_rect = self.canvas.create_rectangle

        if NUMPY_AVAILABLE and isinstance(fb, np.ndarray):
            # Downsample by tiles (draw top-left pixel per tile)
            h, w = fb.shape[0], fb.shape[1]
            for py in range(0, min(240, h), ts):
                row = fb[py]
                py2 = y + py * scale
                for px in range(0, min(320, w), ts):
                    c = row[px]
                    create_rect(
                        x + px * scale, py2,
                        x + (px + ts) * scale, y + (py + ts) * scale,
                        fill=f"#{int(c[0]):02x}{int(c[1]):02x}{int(c[2]):02x}", outline=""
                    )
        else:
            # list-of-tuples fallback
            for py in range(0, 240, ts):
                py2 = y + py * scale
                row = fb[py]
                for px in range(0, 320, ts):
                    r, g, b = row[px]
                    create_rect(
                        x + px * scale, py2,
                        x + (px + ts) * scale, y + (py + ts) * scale,
                        fill=f"#{r:02x}{g:02x}{b:02x}", outline=""
                    )


# ===========================================================================
# main
# ===========================================================================
def main():
    print("=" * 74)
    print("MIPSEMU 2.6 — Enhanced Boot + Fast-Path (Optimized)")
    print("=" * 74)
    print("\nKey Improvements:")
    print("  ✓ Correct delay-slot semantics (retained)")
    print("  ✓ Fixed SP mirror @ 0xA4000000 (real instruction fetch path)")
    print("  ✓ Endianness normalization kept")
    print("  ✓ Faster memory I/O (no struct slicing; fast-path regions)")
    print("  ✓ Fewer attr lookups in hot path")
    print("  ✓ Smoother ~60 FPS pacing (perf_counter_ns & adaptive IPF)")
    print("  ✓ Coarser canvas tiles by default (fewer Tk ops)")
    print("  ✓ PJ64Bridge seam (ctypes) — plug a native core later")
    print("=" * 74)
    print()

    root = tk.Tk()
    app = N64Emulator(root, engine_mode=ENGINE)
    root.mainloop()


if __name__ == "__main__":
    main()
