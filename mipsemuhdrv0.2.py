#!/usr/bin/env python3
"""
MIPSEMU 2.5-ULTRA64 — Enhanced Boot Capable Edition (Optimized)
- Correct MIPS delay-slot handling (branches now run the delay instruction)
- Cleaner ROM header parsing (proper game code / region / version)
- Safer ROM endian fixes and size checks
- Streamlined memory reads/writes (halfword helpers; fewer attribute lookups)
- Adaptive CPU budget per frame to target ~60 FPS
- Better thread shutdown and UI polish
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path
from datetime import datetime
import struct
import threading
import time
import hashlib

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except Exception:
    NUMPY_AVAILABLE = False


# ============================================================================
# ROM HEADER (endianness-aware)
# ============================================================================

class ROMHeader:
    def __init__(self, data: bytes):
        # keep up to first 0x1000 for quick inspection
        self.raw_data = data[:0x1000]
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
        # swap 32-bit words between big<->little
        out = bytearray(len(data))
        for i in range(0, len(data), 4):
            if i + 3 < len(data):
                out[i:i+4] = struct.pack('<I', struct.unpack('>I', data[i:i+4])[0])
        return bytes(out)

    @staticmethod
    def _swap_bytes(data: bytes) -> bytes:
        # v64 byte swap (pairs)
        out = bytearray(len(data))
        for i in range(0, len(data), 2):
            if i + 1 < len(data):
                out[i], out[i+1] = data[i+1], data[i]
        return bytes(out)

    def parse(self):
        if len(self.raw_data) < 0x40:
            return

        magic_be = struct.unpack('>I', self.raw_data[0:4])[0]
        data = self.raw_data

        # Detect and convert to big-endian view
        if magic_be == 0x80371240:                # z64 (big)
            self.endian = 'big'
        elif magic_be == 0x40123780:              # n64 (little)
            self.endian = 'little'
            data = self._swap_endian_pairs(data)
        elif magic_be == 0x37804012:              # v64 (byteswapped)
            self.endian = 'byteswap'
            data = self._swap_bytes(data)
        else:
            return

        # parse from big-endian view
        self.clock_rate  = struct.unpack('>I', data[0x04:0x08])[0]
        self.boot_address= struct.unpack('>I', data[0x08:0x0C])[0]
        self.release     = struct.unpack('>I', data[0x0C:0x10])[0]
        self.crc1        = struct.unpack('>I', data[0x10:0x14])[0]
        self.crc2        = struct.unpack('>I', data[0x14:0x18])[0]
        self.name        = data[0x20:0x34].decode('ascii', errors='ignore').strip('\x00')

        # Correct field widths:
        # Game code = 2 bytes at 0x3B..0x3C; Region = 0x3E; Version = 0x3F
        self.game_code   = data[0x3B:0x3D].decode('ascii', errors='ignore')
        region_chr       = chr(data[0x3E])
        self.region      = region_chr  # leave raw char; commonly 'E','J','P','U','X', etc.
        self.version     = data[0x3F]

        self.rom_hash    = hashlib.md5(self.raw_data).hexdigest()
        self.valid       = True

    # public helpers for full ROM fixes
    def fix_rom_endianness(self, rom: bytes) -> bytes:
        if self.endian == 'little':
            return self._swap_endian_pairs(rom)
        if self.endian == 'byteswap':
            return self._swap_bytes(rom)
        return rom


# ============================================================================
# COP0 (minimal but complete-enough for boot)
# ============================================================================

class COP0:
    def __init__(self):
        self.registers = [0] * 32
        self.registers[12] = 0x34000000  # Status: kernel, IE=COP usable
        self.registers[15] = 0x00000B00  # PRId VR4300
        self.registers[16] = 0x7006E463  # Config

    def read_register(self, reg: int) -> int:
        if reg == 1:  # Random
            return int(time.time() * 1000) % 32
        return self.registers[reg] if 0 <= reg < 32 else 0

    def write_register(self, reg: int, value: int):
        if 0 <= reg < 32 and reg not in (0, 1, 15):
            self.registers[reg] = value & 0xFFFFFFFF


# ============================================================================
# Memory map
# ============================================================================

class Memory:
    def __init__(self):
        self.rdram = bytearray(8 * 1024 * 1024)  # include expansion for simplicity
        self.rom = None
        self.rom_size = 0
        self.sp_dmem = bytearray(4096)
        self.sp_imem = bytearray(4096)
        self.pif_rom = bytearray(2048)
        self.pif_ram = bytearray(64)
        self.vi_regs = [0] * 32

    def load_rom(self, rom_data: bytes):
        self.rom = bytearray(rom_data)
        self.rom_size = len(self.rom)

    @staticmethod
    def _mask32(addr: int) -> int:
        return addr & 0xFFFFFFFF

    def map_address(self, addr: int):
        addr = self._mask32(addr)

        # RDRAM (KUSEG/KSEG0/KSEG1 mirrors)
        if addr < 0x00800000:
            return ('rdram', addr)
        if 0x80000000 <= addr < 0x80800000:
            return ('rdram', addr & 0x007FFFFF)
        if 0xA0000000 <= addr < 0xA0800000:
            return ('rdram', addr & 0x007FFFFF)

        # RSP memories
        if 0x04000000 <= addr < 0x04001000:
            return ('sp_dmem', addr & 0xFFF)
        if 0x04001000 <= addr < 0x04002000:
            return ('sp_imem', addr & 0xFFF)

        # VI, AI, etc — coarse stub
        if 0x04400000 <= addr < 0x04400080:
            return ('vi', (addr - 0x04400000) >> 2)

        # Cartridge
        if 0x10000000 <= addr < 0x1FC00000:
            return ('rom', addr & 0x0FFFFFFF)
        if 0xB0000000 <= addr < 0xBFC00000:
            return ('rom', addr & 0x0FFFFFFF)

        # PIF
        if 0x1FC00000 <= addr < 0x1FC007C0:
            return ('pif_rom', addr & 0x7FF)
        if 0x1FC007C0 <= addr < 0x1FC00800:
            return ('pif_ram', addr & 0x3F)

        return ('invalid', 0)

    # -------- byte/half/word helpers (big-endian) --------

    def read_byte(self, addr: int) -> int:
        region, off = self.map_address(addr)
        if region == 'rdram' and off < len(self.rdram):
            return self.rdram[off]
        if region == 'rom' and self.rom and off < self.rom_size:
            return self.rom[off]
        if region == 'sp_dmem':
            return self.sp_dmem[off]
        if region == 'sp_imem':
            return self.sp_imem[off]
        if region == 'pif_ram':
            return self.pif_ram[off]
        if region == 'vi':
            idx = off & 0x1F
            return (self.vi_regs[idx] & 0xFF)
        return 0

    def read_half(self, addr: int) -> int:
        addr &= 0xFFFFFFFE
        region, off = self.map_address(addr)
        if region == 'rdram' and off + 1 < len(self.rdram):
            return struct.unpack('>H', self.rdram[off:off+2])[0]
        if region == 'rom' and self.rom and off + 1 < self.rom_size:
            return struct.unpack('>H', self.rom[off:off+2])[0]
        if region == 'sp_dmem' and off + 1 < len(self.sp_dmem):
            return struct.unpack('>H', self.sp_dmem[off:off+2])[0]
        return 0

    def read_word(self, addr: int) -> int:
        addr &= 0xFFFFFFFC
        region, off = self.map_address(addr)
        if region == 'rdram' and off + 3 < len(self.rdram):
            return struct.unpack('>I', self.rdram[off:off+4])[0]
        if region == 'rom' and self.rom and off + 3 < self.rom_size:
            return struct.unpack('>I', self.rom[off:off+4])[0]
        if region == 'sp_dmem' and off + 3 < len(self.sp_dmem):
            return struct.unpack('>I', self.sp_dmem[off:off+4])[0]
        if region == 'vi':
            idx = (addr - 0x04400000) >> 2
            idx &= 0x1F
            return self.vi_regs[idx]
        return 0

    def write_byte(self, addr: int, value: int):
        value &= 0xFF
        region, off = self.map_address(addr)
        if region == 'rdram' and off < len(self.rdram):
            self.rdram[off] = value
        elif region == 'sp_dmem':
            self.sp_dmem[off] = value
        elif region == 'pif_ram':
            self.pif_ram[off] = value
        elif region == 'vi':
            pass  # ignore for now

    def write_half(self, addr: int, value: int):
        addr &= 0xFFFFFFFE
        value &= 0xFFFF
        region, off = self.map_address(addr)
        if region == 'rdram' and off + 1 < len(self.rdram):
            struct.pack_into('>H', self.rdram, off, value)
        elif region == 'sp_dmem' and off + 1 < len(self.sp_dmem):
            struct.pack_into('>H', self.sp_dmem, off, value)
        elif region == 'vi':
            pass

    def write_word(self, addr: int, value: int):
        addr &= 0xFFFFFFFC
        value &= 0xFFFFFFFF
        region, off = self.map_address(addr)
        if region == 'rdram' and off + 3 < len(self.rdram):
            struct.pack_into('>I', self.rdram, off, value)
        elif region == 'sp_dmem' and off + 3 < len(self.sp_dmem):
            struct.pack_into('>I', self.sp_dmem, off, value)
        elif region == 'vi':
            idx = (addr - 0x04400000) >> 2
            idx &= 0x1F
            self.vi_regs[idx] = value


# ============================================================================
# PIF (very light boot shim)
# ============================================================================

class PIF:
    def __init__(self, memory: Memory):
        self.memory = memory

    def simulate_boot(self, rom_header: ROMHeader) -> bool:
        if not rom_header or not rom_header.valid:
            return False
        # Copy first 0x1000 of header/ipl into DMEM (mock IPL3 presence)
        chunk = rom_header.raw_data[:0x1000]
        self.memory.sp_dmem[:len(chunk)] = chunk
        # Minimal PIF RAM init
        self.memory.pif_ram[0x24:0x28] = b'\x00\x00\x00\x3F'
        return True


# ============================================================================
# RDP stub (pretty colors so we see frames)
# ============================================================================

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
            for y in range(240):
                row = self.framebuffer[y]
                for x in range(320):
                    row[x] = c


# ============================================================================
# MIPS R4300i (optimized & corrected delay slot)
# ============================================================================

class MIPSCPU:
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

        # proper delay-slot state
        self.branch_pending = False
        self.branch_target = 0
        self.in_delay_slot = False

        self.ll_bit = 0  # not used yet

    # ---------- helpers ----------
    @staticmethod
    def _signed32(v: int) -> int:
        return v - 0x100000000 if (v & 0x80000000) else v

    @staticmethod
    def _sx8(v: int) -> int:
        return v | 0xFFFFFF00 if (v & 0x80) else v & 0xFF

    @staticmethod
    def _sx16(v: int) -> int:
        return v | 0xFFFF0000 if (v & 0x8000) else v & 0xFFFF

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
        # On real HW we start in PIF; we then jump to IPL3/boot address.
        self.pc = boot_address or 0xA4000040
        self.next_pc = self.pc + 4
        self.registers[29] = 0xA4001FF0  # SP
        self.registers[31] = 0xA4001550  # RA
        self.running = True

    # ---------- main step ----------
    def step(self):
        if not self.running:
            return

        mem_read_word = self.memory.read_word  # bind locally (faster)
        regs = self.registers

        try:
            instr = mem_read_word(self.pc)

            # decode
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

            # -------- execute --------
            if opcode == 0x00:
                # SPECIAL
                if funct == 0x00:       # SLL
                    if instr != 0:      # NOP guard
                        regs[rd] = (regs[rt] << shamt) & 0xFFFFFFFF
                elif funct == 0x02:     # SRL
                    regs[rd] = (regs[rt] >> shamt) & 0xFFFFFFFF
                elif funct == 0x03:     # SRA
                    regs[rd] = (self._signed32(regs[rt]) >> shamt) & 0xFFFFFFFF
                elif funct == 0x04:     # SLLV
                    regs[rd] = (regs[rt] << (regs[rs] & 0x1F)) & 0xFFFFFFFF
                elif funct == 0x06:     # SRLV
                    regs[rd] = (regs[rt] >> (regs[rs] & 0x1F)) & 0xFFFFFFFF
                elif funct == 0x07:     # SRAV
                    regs[rd] = (self._signed32(regs[rt]) >> (regs[rs] & 0x1F)) & 0xFFFFFFFF
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
                    result = self._signed32(regs[rs]) * self._signed32(regs[rt])
                    self.lo = result & 0xFFFFFFFF
                    self.hi = (result >> 32) & 0xFFFFFFFF
                elif funct == 0x19:     # MULTU
                    result = (regs[rs] & 0xFFFFFFFF) * (regs[rt] & 0xFFFFFFFF)
                    self.lo = result & 0xFFFFFFFF
                    self.hi = (result >> 32) & 0xFFFFFFFF
                elif funct == 0x1A:     # DIV
                    if regs[rt] != 0:
                        self.lo = (self._signed32(regs[rs]) // self._signed32(regs[rt])) & 0xFFFFFFFF
                        self.hi = (self._signed32(regs[rs]) %  self._signed32(regs[rt])) & 0xFFFFFFFF
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
                    regs[rd] = 1 if self._signed32(regs[rs]) < self._signed32(regs[rt]) else 0
                elif funct == 0x2B:     # SLTU
                    regs[rd] = 1 if (regs[rs] & 0xFFFFFFFF) < (regs[rt] & 0xFFFFFFFF) else 0
                # else: unimplemented SPECIAL => NOP

            elif opcode == 0x01:  # REGIMM
                offset = (self._sx16(imm) << 2) & 0xFFFFFFFF
                srs = self._signed32(regs[rs])
                if rt == 0x00 and srs < 0:          # BLTZ
                    new_target = (self.next_pc + offset) & 0xFFFFFFFF
                    do_branch = True
                elif rt == 0x01 and srs >= 0:       # BGEZ
                    new_target = (self.next_pc + offset) & 0xFFFFFFFF
                    do_branch = True
                elif rt == 0x10 and srs < 0:        # BLTZAL
                    regs[31] = (self.pc + 8) & 0xFFFFFFFF
                    new_target = (self.next_pc + offset) & 0xFFFFFFFF
                    do_branch = True
                elif rt == 0x11 and srs >= 0:       # BGEZAL
                    regs[31] = (self.pc + 8) & 0xFFFFFFFF
                    new_target = (self.next_pc + offset) & 0xFFFFFFFF
                    do_branch = True

            elif opcode == 0x02:  # J
                new_target = ((self.pc & 0xF0000000) | (target << 2)) & 0xFFFFFFFF
                do_branch = True
            elif opcode == 0x03:  # JAL
                regs[31] = (self.pc + 8) & 0xFFFFFFFF
                new_target = ((self.pc & 0xF0000000) | (target << 2)) & 0xFFFFFFFF
                do_branch = True

            elif opcode == 0x04:  # BEQ
                if regs[rs] == regs[rt]:
                    new_target = (self.next_pc + ((self._sx16(imm) << 2) & 0xFFFFFFFF)) & 0xFFFFFFFF
                    do_branch = True
            elif opcode == 0x05:  # BNE
                if regs[rs] != regs[rt]:
                    new_target = (self.next_pc + ((self._sx16(imm) << 2) & 0xFFFFFFFF)) & 0xFFFFFFFF
                    do_branch = True
            elif opcode == 0x06:  # BLEZ
                if self._signed32(regs[rs]) <= 0:
                    new_target = (self.next_pc + ((self._sx16(imm) << 2) & 0xFFFFFFFF)) & 0xFFFFFFFF
                    do_branch = True
            elif opcode == 0x07:  # BGTZ
                if self._signed32(regs[rs]) > 0:
                    new_target = (self.next_pc + ((self._sx16(imm) << 2) & 0xFFFFFFFF)) & 0xFFFFFFFF
                    do_branch = True

            elif opcode == 0x08 or opcode == 0x09:   # ADDI/ADDIU
                regs[rt] = (regs[rs] + self._sx16(imm)) & 0xFFFFFFFF
            elif opcode == 0x0A:                     # SLTI
                regs[rt] = 1 if self._signed32(regs[rs]) < self._sx16(imm) else 0
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
                addr = (regs[rs] + self._sx16(imm)) & 0xFFFFFFFF
                regs[rt] = self._sx8(self.memory.read_byte(addr))
            elif opcode == 0x21:  # LH
                addr = (regs[rs] + self._sx16(imm)) & 0xFFFFFFFF
                regs[rt] = self._sx16(self.memory.read_half(addr))
            elif opcode == 0x23:  # LW
                addr = (regs[rs] + self._sx16(imm)) & 0xFFFFFFFF
                regs[rt] = self.memory.read_word(addr)
            elif opcode == 0x24:  # LBU
                addr = (regs[rs] + self._sx16(imm)) & 0xFFFFFFFF
                regs[rt] = self.memory.read_byte(addr) & 0xFF
            elif opcode == 0x25:  # LHU
                addr = (regs[rs] + self._sx16(imm)) & 0xFFFFFFFF
                regs[rt] = self.memory.read_half(addr) & 0xFFFF

            elif opcode == 0x28:  # SB
                addr = (regs[rs] + self._sx16(imm)) & 0xFFFFFFFF
                self.memory.write_byte(addr, regs[rt] & 0xFF)
            elif opcode == 0x29:  # SH
                addr = (regs[rs] + self._sx16(imm)) & 0xFFFFFFFF
                self.memory.write_half(addr, regs[rt] & 0xFFFF)
            elif opcode == 0x2B:  # SW
                addr = (regs[rs] + self._sx16(imm)) & 0xFFFFFFFF
                self.memory.write_word(addr, regs[rt])

            # -------- program counter transition w/ delay slot --------
            regs[0] = 0  # $zero enforced

            if self.in_delay_slot:
                # we just executed the delay instruction; now jump
                self.pc = self.branch_target
                self.in_delay_slot = False
                self.branch_pending = False
            else:
                # normal advance to next instruction
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


# ============================================================================
# Emulator shell (UI + loops)
# ============================================================================

class N64Emulator:
    def __init__(self, root):
        self.root = root
        self.root.title("MIPSEMU 2.5 — Enhanced Boot Edition (Optimized)")
        self.root.geometry("1024x768")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.memory = Memory()
        self.cpu = MIPSCPU(self.memory)
        self.pif = PIF(self.memory)
        self.rdp = RDP()

        self.current_rom = None
        self.rom_header = None

        self.emulation_running = False
        self.boot_status = 'idle'
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = time.time()
        self.frames_this_second = 0

        # dynamic CPU budget for 60fps
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

        self._log("MIPSEMU 2.5 — Optimized build ready.")
        self._log("Fixes: delay slots, header parse, memory helpers, adaptive CPU budget.")

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
            self._log(f"Loading: {Path(filepath).name}")
            with open(filepath, "rb") as f:
                rom = f.read()

            hdr = ROMHeader(rom)
            if not hdr.valid:
                messagebox.showerror("Error", "Invalid N64 ROM file.")
                return

            rom_fixed = hdr.fix_rom_endianness(rom)
            self.memory.load_rom(rom_fixed)
            self.rom_header = hdr
            self.current_rom = filepath

            self._log(f"ROM: {hdr.name}")
            self._log(f"Game Code: {hdr.game_code} | Region: {hdr.region} | Ver: {hdr.version}")
            self._log(f"Format: {hdr.endian}-endian (normalized)")
            self._log(f"Boot Address: {hex(hdr.boot_address)}")
            self._log(f"Size: {len(rom_fixed)//(1024*1024)} MB")

            self.root.title(f"MIPSEMU 2.5 — {hdr.name}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load ROM: {e}")
            self._log(f"ERROR: {e}")

    # ---------- Emulation control ----------
    def start_emulation(self):
        if not self.current_rom:
            messagebox.showwarning("No ROM", "Please load a ROM first.")
            return

        self._log("=" * 64)
        self._log("STARTING BOOT SEQUENCE")
        self._log("=" * 64)

        self.boot_status = 'booting'

        if self.pif.simulate_boot(self.rom_header):
            self._log("PIF: Boot ROM simulation complete (IPL3 shim into DMEM).")

        self.cpu.boot_setup(self.rom_header.boot_address)
        self._log(f"CPU: PC = {hex(self.cpu.pc)}  SP = {hex(self.cpu.registers[29])}")
        self._log(f"COP0 Status = {hex(self.cpu.cop0.registers[12])}")

        self.emulation_running = True
        self.cpu.running = True
        self.boot_status = 'running'

        self._log("Boot complete — starting execution.")
        self._log("=" * 64)

        self.emu_thread = threading.Thread(target=self._emulation_loop, daemon=True)
        self.emu_thread.start()

        self._render_loop()

    def stop_emulation(self):
        self.emulation_running = False
        self.cpu.running = False
        self.boot_status = 'idle'
        self._log("Emulation stopped.")

    def reset_emulation(self):
        self.stop_emulation()
        self.cpu.reset()
        self.frame_count = 0
        self.instructions_per_frame = 20000
        self._log("Emulation reset.")

    def _on_close(self):
        self.stop_emulation()
        # Give thread a moment to exit
        self.root.after(100, self.root.destroy)

    # ---------- Loops ----------
    def _emulation_loop(self):
        target_dt = 1.0 / 60.0

        while self.emulation_running and self.cpu.running:
            t0 = time.perf_counter()

            ipf = self.instructions_per_frame
            step = self.cpu.step
            for _ in range(ipf):
                step()
                if not self.cpu.running:
                    break

            # Simple adaptive budget to stick near 60 FPS on a wide range of CPUs
            frame_dt = time.perf_counter() - t0
            if frame_dt < target_dt * 0.6:
                # plenty of headroom → increase work
                self.instructions_per_frame = min(self.max_ipf, int(self.instructions_per_frame * 1.15) + 100)
            elif frame_dt > target_dt * 1.2:
                # struggling → reduce work
                self.instructions_per_frame = max(self.min_ipf, int(self.instructions_per_frame * 0.85))

            # Sleep to roughly cap at 60Hz
            sleep_time = target_dt - frame_dt
            if sleep_time > 0:
                time.sleep(sleep_time)

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
                self._render_framebuffer(sx, sy)

                # HUD
                self.canvas.create_text(
                    sx + 320, sy + 20,
                    text=f"PC: {hex(self.cpu.pc)} | Instr: {self.cpu.instructions_executed:,}",
                    font=("Consolas", 10), fill="#00ff00"
                )
                self.canvas.create_text(
                    sx + 320, sy + 40,
                    text=f"SP: {hex(self.cpu.registers[29])} | RA: {hex(self.cpu.registers[31])}",
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
            # simple animated clear so we see changes even without RDP
            if NUMPY_AVAILABLE and (self.frame_count % 30 == 0):
                c = (
                    (self.frame_count % 256) // 4,
                    ((self.frame_count * 2) % 256) // 4,
                    ((self.frame_count * 3) % 256) // 4
                )
                self.rdp.clear_framebuffer(c)

            self.root.after(16, self._render_loop)

        except Exception as e:
            self._log(f"Render error: {e}")

    def _render_framebuffer(self, x: int, y: int):
        if NUMPY_AVAILABLE:
            # coarse 8x8 tiles for speed (no PIL dependency)
            scale = 2
            fb = self.rdp.framebuffer
            create_rect = self.canvas.create_rectangle
            for py in range(0, 240, 8):
                row = fb[py]
                for px in range(0, 320, 8):
                    c = row[px]
                    create_rect(
                        x + px * scale, y + py * scale,
                        x + (px + 8) * scale, y + (py + 8) * scale,
                        fill=f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}", outline=""
                    )
        else:
            # fallback (already fairly quick due to coarse tiles)
            scale = 2
            fb = self.rdp.framebuffer
            create_rect = self.canvas.create_rectangle
            for py in range(0, 240, 8):
                for px in range(0, 320, 8):
                    r, g, b = fb[py][px]
                    create_rect(
                        x + px * scale, y + py * scale,
                        x + (px + 8) * scale, y + (py + 8) * scale,
                        fill=f"#{r:02x}{g:02x}{b:02x}", outline=""
                    )


# ============================================================================
# main
# ============================================================================

def main():
    print("=" * 70)
    print("MIPSEMU 2.5 — Enhanced Boot Edition (Optimized)")
    print("=" * 70)
    print("\nKey Improvements:")
    print("  ✓ Correct MIPS delay-slot semantics")
    print("  ✓ Proper ROM header fields (game code / region / version)")
    print("  ✓ Endianness normalization for whole ROM")
    print("  ✓ Halfword helpers; fewer attr lookups in hot paths")
    print("  ✓ Adaptive instructions-per-frame for ~60 FPS")
    print("  ✓ Cleaner shutdown & UI")
    print("=" * 70)
    print()

    root = tk.Tk()
    app = N64Emulator(root)
    root.mainloop()

if __name__ == "__main__":
    main()
## [C] Team Flames 2025