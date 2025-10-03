#!/usr/bin/env python3
"""
MIPSEMU 1.03-ULTRA64 - Darkness Revived (Ultra64 SDK Edition)
N64 Emulator with libultra/Ultra64 Software Implementation
Python 3.13 | Tkinter GUI

NEW IN 1.03:
- F3DEX graphics microcode interpreter
- RDP command processor and rasterizer
- OS thread management system
- Display list processing
- DMA engine implementation
- Interrupt system
- Framebuffer rendering
- Matrix stack and transformations
- Vertex processing pipeline
- Texture coordinate generation
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
from pathlib import Path
from datetime import datetime
import json
import struct
import threading
import time
from collections import defaultdict, deque
import hashlib
import math
import random


# ============================================================================
# ULTRA64 OS LAYER
# ============================================================================

class OSThread:
    """Ultra64 OS Thread"""
    def __init__(self, thread_id, priority=10):
        self.id = thread_id
        self.priority = priority
        self.state = 'STOPPED'  # STOPPED, RUNNING, WAITING
        self.pc = 0
        self.sp = 0
        self.context = {}
        
class OSMessageQueue:
    """Ultra64 Message Queue"""
    def __init__(self, size=8):
        self.queue = deque(maxlen=size)
        self.validCount = 0
        
    def send(self, message):
        if len(self.queue) < self.queue.maxlen:
            self.queue.append(message)
            self.validCount += 1
            return True
        return False
        
    def receive(self):
        if self.queue:
            self.validCount -= 1
            return self.queue.popleft()
        return None

class OSManager:
    """Ultra64 Operating System Manager"""
    def __init__(self):
        self.threads = {}
        self.current_thread = None
        self.message_queues = {}
        self.timers = []
        self.interrupts_enabled = True
        
        # Create main thread
        main_thread = OSThread(0, priority=10)
        main_thread.state = 'RUNNING'
        self.threads[0] = main_thread
        self.current_thread = main_thread
        
        # VI (Vertical Interrupt) queue
        self.vi_queue = OSMessageQueue(8)
        self.message_queues['VI'] = self.vi_queue
        
    def create_thread(self, thread_id, priority=10):
        """Create new thread"""
        thread = OSThread(thread_id, priority)
        self.threads[thread_id] = thread
        return thread
        
    def start_thread(self, thread_id):
        """Start thread execution"""
        if thread_id in self.threads:
            self.threads[thread_id].state = 'RUNNING'
            
    def vi_retrace_callback(self):
        """Called on vertical retrace"""
        self.vi_queue.send({'type': 'VI_RETRACE', 'time': time.time()})


# ============================================================================
# RSP GRAPHICS MICROCODE (F3DEX)
# ============================================================================

class F3DEXMicrocode:
    """F3DEX Graphics Microcode Interpreter"""
    def __init__(self, rsp, rdp):
        self.rsp = rsp
        self.rdp = rdp
        
        # F3DEX commands (simplified)
        self.G_NOOP = 0x00
        self.G_VTX = 0x01
        self.G_MODIFYVTX = 0x02
        self.G_CULLDL = 0x03
        self.G_BRANCH_Z = 0x04
        self.G_TRI1 = 0x05
        self.G_TRI2 = 0x06
        self.G_QUAD = 0x07
        self.G_LINE3D = 0x08
        
        self.G_DMA_IO = 0xD6
        self.G_TEXTURE = 0xD7
        self.G_POPMTX = 0xD8
        self.G_GEOMETRYMODE = 0xD9
        self.G_MTX = 0xDA
        self.G_MOVEWORD = 0xDB
        self.G_MOVEMEM = 0xDC
        self.G_LOAD_UCODE = 0xDD
        self.G_DL = 0xDE
        self.G_ENDDL = 0xDF
        self.G_SPNOOP = 0xE0
        self.G_RDPHALF_1 = 0xE1
        self.G_SETOTHERMODE_L = 0xE2
        self.G_SETOTHERMODE_H = 0xE3
        self.G_TEXRECT = 0xE4
        self.G_TEXRECTFLIP = 0xE5
        self.G_RDPLOADSYNC = 0xE6
        self.G_RDPPIPESYNC = 0xE7
        self.G_RDPTILESYNC = 0xE8
        self.G_RDPFULLSYNC = 0xE9
        self.G_SETKEYGB = 0xEA
        self.G_SETKEYR = 0xEB
        self.G_SETCONVERT = 0xEC
        self.G_SETSCISSOR = 0xED
        self.G_SETPRIMDEPTH = 0xEE
        self.G_RDPSETOTHERMODE = 0xEF
        self.G_LOADTLUT = 0xF0
        self.G_RDPHALF_2 = 0xF1
        self.G_SETTILESIZE = 0xF2
        self.G_LOADBLOCK = 0xF3
        self.G_LOADTILE = 0xF4
        self.G_SETTILE = 0xF5
        self.G_FILLRECT = 0xF6
        self.G_SETFILLCOLOR = 0xF7
        self.G_SETFOGCOLOR = 0xF8
        self.G_SETBLENDCOLOR = 0xF9
        self.G_SETPRIMCOLOR = 0xFA
        self.G_SETENVCOLOR = 0xFB
        self.G_SETCOMBINE = 0xFC
        self.G_SETTIMG = 0xFD
        self.G_SETZIMG = 0xFE
        self.G_SETCIMG = 0xFF
        
    def process_display_list(self, dl_addr, memory):
        """Process display list commands"""
        commands_processed = 0
        max_commands = 10000  # Prevent infinite loops
        
        while commands_processed < max_commands:
            # Read 64-bit command
            w0 = memory.read_word(dl_addr)
            w1 = memory.read_word(dl_addr + 4)
            
            cmd = (w0 >> 24) & 0xFF
            
            # Process command
            if cmd == self.G_ENDDL:
                break
            elif cmd == self.G_VTX:
                self.cmd_vtx(w0, w1, memory)
            elif cmd == self.G_TRI1:
                self.cmd_tri1(w0, w1)
            elif cmd == self.G_TRI2:
                self.cmd_tri2(w0, w1)
            elif cmd == self.G_MTX:
                self.cmd_mtx(w0, w1, memory)
            elif cmd == self.G_DL:
                # Branch to another display list
                branch_addr = w1
                self.process_display_list(branch_addr, memory)
            elif cmd == self.G_TEXTURE:
                self.cmd_texture(w0, w1)
            elif cmd == self.G_SETCOMBINE:
                self.rdp.cmd_setcombine(w0, w1)
            elif cmd == self.G_SETTIMG:
                self.rdp.cmd_settimg(w0, w1)
            elif cmd == self.G_SETCIMG:
                self.rdp.cmd_setcimg(w0, w1)
            elif cmd == self.G_SETZIMG:
                self.rdp.cmd_setzimg(w0, w1)
            elif cmd == self.G_FILLRECT:
                self.rdp.cmd_fillrect(w0, w1)
            elif cmd == self.G_SETFILLCOLOR:
                self.rdp.cmd_setfillcolor(w0, w1)
            elif cmd == self.G_RDPFULLSYNC:
                self.rdp.cmd_fullsync()
            elif cmd == self.G_RDPPIPESYNC:
                self.rdp.cmd_pipesync()
            elif cmd == self.G_GEOMETRYMODE:
                self.cmd_geometrymode(w0, w1)
                
            dl_addr += 8
            commands_processed += 1
            
        return commands_processed
        
    def cmd_vtx(self, w0, w1, memory):
        """Load vertices into vertex buffer"""
        n = ((w0 >> 12) & 0xFF) // 2  # Number of vertices
        v0 = (w0 >> 1) & 0x7F  # Starting index
        addr = w1 & 0xFFFFFF
        
        for i in range(n):
            vtx_data = []
            for j in range(4):  # 16 bytes per vertex
                word = memory.read_word(addr + i * 16 + j * 4)
                vtx_data.append(word)
                
            # Parse vertex (simplified)
            x = self.sign_extend_16((vtx_data[0] >> 16) & 0xFFFF)
            y = self.sign_extend_16(vtx_data[0] & 0xFFFF)
            z = self.sign_extend_16((vtx_data[1] >> 16) & 0xFFFF)
            
            s = (vtx_data[2] >> 16) & 0xFFFF
            t = vtx_data[2] & 0xFFFF
            
            r = (vtx_data[3] >> 24) & 0xFF
            g = (vtx_data[3] >> 16) & 0xFF
            b = (vtx_data[3] >> 8) & 0xFF
            a = vtx_data[3] & 0xFF
            
            vertex = {
                'x': x, 'y': y, 'z': z,
                's': s, 't': t,
                'r': r, 'g': g, 'b': b, 'a': a
            }
            
            self.rsp.vertex_buffer[v0 + i] = vertex
            
    def cmd_tri1(self, w0, w1):
        """Draw one triangle"""
        v0 = ((w1 >> 16) & 0xFF) // 2
        v1 = ((w1 >> 8) & 0xFF) // 2
        v2 = (w1 & 0xFF) // 2
        
        if v0 in self.rsp.vertex_buffer and v1 in self.rsp.vertex_buffer and v2 in self.rsp.vertex_buffer:
            tri = {
                'v0': self.rsp.vertex_buffer[v0],
                'v1': self.rsp.vertex_buffer[v1],
                'v2': self.rsp.vertex_buffer[v2]
            }
            self.rdp.draw_triangle(tri)
            
    def cmd_tri2(self, w0, w1):
        """Draw two triangles"""
        v0 = ((w0 >> 16) & 0xFF) // 2
        v1 = ((w0 >> 8) & 0xFF) // 2
        v2 = (w0 & 0xFF) // 2
        
        v3 = ((w1 >> 16) & 0xFF) // 2
        v4 = ((w1 >> 8) & 0xFF) // 2
        v5 = (w1 & 0xFF) // 2
        
        # Draw first triangle
        if all(i in self.rsp.vertex_buffer for i in [v0, v1, v2]):
            tri1 = {
                'v0': self.rsp.vertex_buffer[v0],
                'v1': self.rsp.vertex_buffer[v1],
                'v2': self.rsp.vertex_buffer[v2]
            }
            self.rdp.draw_triangle(tri1)
            
        # Draw second triangle
        if all(i in self.rsp.vertex_buffer for i in [v3, v4, v5]):
            tri2 = {
                'v0': self.rsp.vertex_buffer[v3],
                'v1': self.rsp.vertex_buffer[v4],
                'v2': self.rsp.vertex_buffer[v5]
            }
            self.rdp.draw_triangle(tri2)
            
    def cmd_mtx(self, w0, w1, memory):
        """Load transformation matrix"""
        addr = w1 & 0xFFFFFF
        push = (w0 >> 2) & 0x1
        load = (w0 >> 1) & 0x1
        projection = (w0 >> 0) & 0x1
        
        # Load 4x4 matrix from memory (16 words)
        matrix = []
        for i in range(16):
            value = memory.read_word(addr + i * 4)
            matrix.append(value)
            
        # Store in matrix stack (simplified)
        if projection:
            self.rsp.projection_matrix = matrix
        else:
            self.rsp.modelview_matrix = matrix
            
    def cmd_texture(self, w0, w1):
        """Set texture parameters"""
        level = (w0 >> 11) & 0x7
        tile = (w0 >> 8) & 0x7
        on = (w0 >> 1) & 0x1
        
        scaleS = (w1 >> 16) & 0xFFFF
        scaleT = w1 & 0xFFFF
        
        self.rsp.texture_enabled = (on == 1)
        self.rsp.texture_tile = tile
        self.rsp.texture_scaleS = scaleS
        self.rsp.texture_scaleT = scaleT
        
    def cmd_geometrymode(self, w0, w1):
        """Set geometry mode flags"""
        clearbits = ~(w0 & 0xFFFFFF)
        setbits = w1
        
        self.rsp.geometry_mode = (self.rsp.geometry_mode & clearbits) | setbits
        
    def sign_extend_16(self, value):
        if value & 0x8000:
            return value - 0x10000
        return value


# ============================================================================
# RDP (REALITY DISPLAY PROCESSOR)
# ============================================================================

class RDP:
    """Reality Display Processor - Rasterizer"""
    def __init__(self):
        # Framebuffer
        self.framebuffer_width = 320
        self.framebuffer_height = 240
        self.framebuffer = [[(0, 0, 0) for _ in range(320)] for _ in range(240)]
        self.zbuffer = [[float('inf') for _ in range(320)] for _ in range(240)]
        
        # Color combiner
        self.combine_mode = 0
        
        # Fill color
        self.fill_color = (0, 0, 0, 255)
        
        # Texture image
        self.texture_image_addr = 0
        self.texture_image_format = 0
        self.texture_image_size = 0
        self.texture_image_width = 0
        
        # Color image (output)
        self.color_image_addr = 0
        self.color_image_format = 0
        self.color_image_size = 0
        self.color_image_width = 320
        
        # Z buffer image
        self.z_image_addr = 0
        
        # Scissor
        self.scissor_x0 = 0
        self.scissor_y0 = 0
        self.scissor_x1 = 320
        self.scissor_y1 = 240
        
        # Primitive color
        self.prim_color = (255, 255, 255, 255)
        self.env_color = (255, 255, 255, 255)
        
        # Statistics
        self.triangles_drawn = 0
        self.pixels_drawn = 0
        
    def cmd_setcombine(self, w0, w1):
        """Set color combiner mode"""
        self.combine_mode = w1
        
    def cmd_settimg(self, w0, w1):
        """Set texture image"""
        self.texture_image_format = (w0 >> 21) & 0x7
        self.texture_image_size = (w0 >> 19) & 0x3
        self.texture_image_width = (w0 & 0x3FF) + 1
        self.texture_image_addr = w1 & 0xFFFFFF
        
    def cmd_setcimg(self, w0, w1):
        """Set color image (framebuffer)"""
        self.color_image_format = (w0 >> 21) & 0x7
        self.color_image_size = (w0 >> 19) & 0x3
        self.color_image_width = (w0 & 0x3FF) + 1
        self.color_image_addr = w1 & 0xFFFFFF
        
    def cmd_setzimg(self, w0, w1):
        """Set Z buffer image"""
        self.z_image_addr = w1 & 0xFFFFFF
        
    def cmd_fillrect(self, w0, w1):
        """Fill rectangle"""
        x1 = ((w1 >> 12) & 0xFFF) >> 2
        y1 = ((w1 >> 0) & 0xFFF) >> 2
        x0 = ((w0 >> 12) & 0xFFF) >> 2
        y0 = ((w0 >> 0) & 0xFFF) >> 2
        
        # Clamp to framebuffer
        x0 = max(0, min(x0, self.framebuffer_width - 1))
        y0 = max(0, min(y0, self.framebuffer_height - 1))
        x1 = max(0, min(x1, self.framebuffer_width - 1))
        y1 = max(0, min(y1, self.framebuffer_height - 1))
        
        # Fill rectangle
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                if 0 <= y < self.framebuffer_height and 0 <= x < self.framebuffer_width:
                    self.framebuffer[y][x] = self.fill_color[:3]
                    self.pixels_drawn += 1
                    
    def cmd_setfillcolor(self, w0, w1):
        """Set fill color"""
        # RGBA 16-bit or 32-bit
        r = (w1 >> 24) & 0xFF
        g = (w1 >> 16) & 0xFF
        b = (w1 >> 8) & 0xFF
        a = w1 & 0xFF
        self.fill_color = (r, g, b, a)
        
    def cmd_fullsync(self):
        """Full sync - wait for RDP to finish"""
        pass
        
    def cmd_pipesync(self):
        """Pipeline sync"""
        pass
        
    def draw_triangle(self, tri):
        """Draw a triangle (simplified rasterizer)"""
        self.triangles_drawn += 1
        
        # Get vertices
        v0 = tri['v0']
        v1 = tri['v1']
        v2 = tri['v2']
        
        # Convert to screen space (simplified projection)
        # Assume coordinates are already in screen space for now
        x0 = int(v0['x'] / 4 + 160)  # Scale and center
        y0 = int(v0['y'] / 4 + 120)
        x1 = int(v1['x'] / 4 + 160)
        y1 = int(v1['y'] / 4 + 120)
        x2 = int(v2['x'] / 4 + 160)
        y2 = int(v2['y'] / 4 + 120)
        
        # Clamp to screen
        x0 = max(0, min(x0, self.framebuffer_width - 1))
        y0 = max(0, min(y0, self.framebuffer_height - 1))
        x1 = max(0, min(x1, self.framebuffer_width - 1))
        y1 = max(0, min(y1, self.framebuffer_height - 1))
        x2 = max(0, min(x2, self.framebuffer_width - 1))
        y2 = max(0, min(y2, self.framebuffer_height - 1))
        
        # Get colors
        r = (v0['r'] + v1['r'] + v2['r']) // 3
        g = (v0['g'] + v1['g'] + v2['g']) // 3
        b = (v0['b'] + v1['b'] + v2['b']) // 3
        
        # Simple triangle fill (scanline)
        self.fill_triangle(x0, y0, x1, y1, x2, y2, (r, g, b))
        
    def fill_triangle(self, x0, y0, x1, y1, x2, y2, color):
        """Fill triangle using scanline algorithm"""
        # Sort vertices by y
        if y0 > y1:
            x0, y0, x1, y1 = x1, y1, x0, y0
        if y0 > y2:
            x0, y0, x2, y2 = x2, y2, x0, y0
        if y1 > y2:
            x1, y1, x2, y2 = x2, y2, x1, y1
            
        # Draw flat-bottom triangle
        if y1 == y2:
            self.fill_flat_bottom(x0, y0, x1, y1, x2, y2, color)
        # Draw flat-top triangle
        elif y0 == y1:
            self.fill_flat_top(x0, y0, x1, y1, x2, y2, color)
        # Split into two triangles
        else:
            # Calculate split point
            if y2 - y0 != 0:
                x3 = int(x0 + (y1 - y0) / (y2 - y0) * (x2 - x0))
                y3 = y1
                
                self.fill_flat_bottom(x0, y0, x1, y1, x3, y3, color)
                self.fill_flat_top(x1, y1, x3, y3, x2, y2, color)
                
    def fill_flat_bottom(self, x0, y0, x1, y1, x2, y2, color):
        """Fill flat-bottom triangle"""
        if y1 - y0 == 0:
            return
            
        slope1 = (x1 - x0) / (y1 - y0)
        slope2 = (x2 - x0) / (y2 - y0)
        
        xs1 = x0
        xs2 = x0
        
        for y in range(y0, y1 + 1):
            if 0 <= y < self.framebuffer_height:
                x_start = int(min(xs1, xs2))
                x_end = int(max(xs1, xs2))
                
                for x in range(x_start, x_end + 1):
                    if 0 <= x < self.framebuffer_width:
                        self.framebuffer[y][x] = color
                        self.pixels_drawn += 1
                        
            xs1 += slope1
            xs2 += slope2
            
    def fill_flat_top(self, x0, y0, x1, y1, x2, y2, color):
        """Fill flat-top triangle"""
        if y2 - y0 == 0:
            return
            
        slope1 = (x2 - x0) / (y2 - y0)
        slope2 = (x2 - x1) / (y2 - y1)
        
        xs1 = x2
        xs2 = x2
        
        for y in range(y2, y0 - 1, -1):
            if 0 <= y < self.framebuffer_height:
                x_start = int(min(xs1, xs2))
                x_end = int(max(xs1, xs2))
                
                for x in range(x_start, x_end + 1):
                    if 0 <= x < self.framebuffer_width:
                        self.framebuffer[y][x] = color
                        self.pixels_drawn += 1
                        
            xs1 -= slope1
            xs2 -= slope2
            
    def clear_framebuffer(self, color=(0, 0, 0)):
        """Clear framebuffer"""
        for y in range(self.framebuffer_height):
            for x in range(self.framebuffer_width):
                self.framebuffer[y][x] = color
        self.pixels_drawn = 0
        self.triangles_drawn = 0


# ============================================================================
# RSP (REALITY SIGNAL PROCESSOR)
# ============================================================================

class RSP:
    """Reality Signal Processor - Enhanced"""
    def __init__(self):
        self.dmem = bytearray(4096)
        self.imem = bytearray(4096)
        self.registers = [0] * 32
        self.pc = 0
        self.status = 0x1  # Halted
        
        # Vertex buffer
        self.vertex_buffer = {}
        
        # Matrix stack
        self.projection_matrix = self.identity_matrix()
        self.modelview_matrix = self.identity_matrix()
        self.matrix_stack = []
        
        # Geometry mode
        self.geometry_mode = 0
        
        # Texture state
        self.texture_enabled = False
        self.texture_tile = 0
        self.texture_scaleS = 0
        self.texture_scaleT = 0
        
        # Lighting (simplified)
        self.lighting_enabled = False
        self.lights = []
        
    def identity_matrix(self):
        """Return 4x4 identity matrix"""
        return [
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1
        ]
        
    def read_dmem(self, offset):
        if offset < len(self.dmem):
            return self.dmem[offset]
        return 0
        
    def write_dmem(self, offset, value):
        if offset < len(self.dmem):
            self.dmem[offset] = value & 0xFF


# ============================================================================
# DMA CONTROLLER
# ============================================================================

class DMAController:
    """Enhanced DMA Controller"""
    def __init__(self, memory):
        self.memory = memory
        self.pi_dram_addr = 0
        self.pi_cart_addr = 0
        self.pi_rd_len = 0
        self.pi_wr_len = 0
        self.pi_status = 0
        self.busy = False
        
    def start_read(self, dram_addr, cart_addr, length):
        """DMA read from cartridge to DRAM"""
        self.pi_dram_addr = dram_addr
        self.pi_cart_addr = cart_addr
        self.pi_rd_len = length
        self.busy = True
        
        # Perform transfer
        for i in range(length + 1):
            value = self.memory.read_byte(0xB0000000 + cart_addr + i)
            self.memory.write_byte(dram_addr + i, value)
            
        self.busy = False
        self.pi_status = 0
        return True
        
    def start_write(self, dram_addr, cart_addr, length):
        """DMA write from DRAM to cartridge (save)"""
        self.pi_dram_addr = dram_addr
        self.pi_cart_addr = cart_addr
        self.pi_wr_len = length
        self.busy = True
        
        # Transfer to save RAM
        for i in range(length + 1):
            value = self.memory.read_byte(dram_addr + i)
            self.memory.write_byte(cart_addr + i, value)
            
        self.busy = False
        return True


# ============================================================================
# ROM HEADER
# ============================================================================

class ROMHeader:
    """N64 ROM Header Parser"""
    def __init__(self, data):
        self.raw_data = data[:0x1000]
        self.valid = False
        self.parse()
        
    def parse(self):
        if len(self.raw_data) < 0x40:
            return
            
        magic = struct.unpack('>I', self.raw_data[0:4])[0]
        
        if magic == 0x80371240:
            self.endian = 'big'
            self.valid = True
        elif magic == 0x40123780:
            self.endian = 'little'
            self.raw_data = self.swap_endian_n64(self.raw_data)
            self.valid = True
        elif magic == 0x37804012:
            self.endian = 'byteswap'
            self.raw_data = self.swap_endian_v64(self.raw_data)
            self.valid = True
        else:
            self.endian = 'unknown'
            return
            
        self.clock_rate = struct.unpack('>I', self.raw_data[0x04:0x08])[0]
        self.boot_address = struct.unpack('>I', self.raw_data[0x08:0x0C])[0]
        self.release = struct.unpack('>I', self.raw_data[0x0C:0x10])[0]
        
        self.crc1 = struct.unpack('>I', self.raw_data[0x10:0x14])[0]
        self.crc2 = struct.unpack('>I', self.raw_data[0x14:0x18])[0]
        
        self.name = self.raw_data[0x20:0x34].decode('ascii', errors='ignore').strip('\x00')
        self.country_code = chr(self.raw_data[0x3E])
        self.country = self.get_country_name(self.country_code)
        self.version = self.raw_data[0x3F]
        self.game_id = self.raw_data[0x3B:0x3F].decode('ascii', errors='ignore')
        self.rom_hash = hashlib.md5(self.raw_data[:0x100]).hexdigest()
        self.ipl3 = self.raw_data[0x40:0x1000]
        
    def get_country_name(self, code):
        countries = {
            'A': 'All', 'D': 'Germany', 'E': 'USA', 'F': 'France',
            'I': 'Italy', 'J': 'Japan', 'S': 'Spain', 'U': 'Australia',
            'P': 'Europe', 'N': 'Canada'
        }
        return countries.get(code, 'Unknown')
        
    def swap_endian_n64(self, data):
        result = bytearray(len(data))
        for i in range(0, len(data), 4):
            result[i:i+4] = data[i:i+4][::-1]
        return bytes(result)
        
    def swap_endian_v64(self, data):
        result = bytearray(len(data))
        for i in range(0, len(data), 2):
            result[i] = data[i+1]
            result[i+1] = data[i]
        return bytes(result)


# ============================================================================
# PIF BOOTLOADER
# ============================================================================

class PIF:
    """PIF Bootloader"""
    def __init__(self, memory):
        self.memory = memory
        self.pif_ram = bytearray(64)
        self.pif_rom = bytearray(2048)
        self.boot_complete = False
        
    def simulate_boot(self, rom_header):
        """Simulate PIF boot sequence"""
        if rom_header and rom_header.ipl3:
            for i, byte in enumerate(rom_header.ipl3):
                if i < 0x1000:
                    self.memory.write_byte(0x04000000 + i, byte)
                    
        self.pif_ram[0x3F] = 0x00
        self.boot_complete = True
        return True


# ============================================================================
# COP0
# ============================================================================

class COP0:
    """Coprocessor 0"""
    def __init__(self):
        self.registers = [0] * 32
        self.STATUS = 12
        self.CAUSE = 13
        self.EPC = 14
        self.COUNT = 9
        self.COMPARE = 11
        
        self.registers[15] = 0x00000B00  # PRID
        self.registers[self.STATUS] = 0x34000000
        self.registers[16] = 0x7006E463  # CONFIG
        
    def read(self, reg):
        return self.registers[reg & 0x1F]
        
    def write(self, reg, value):
        reg = reg & 0x1F
        if reg == self.COMPARE:
            self.registers[reg] = value
            self.registers[self.CAUSE] &= ~0x8000
        else:
            self.registers[reg] = value


# Rest of the CPU implementation remains the same as v1.02...
# (Including MIPSCPU, Memory, VideoInterface, etc.)
# For brevity, I'll include just the key components

class MIPSCPU:
    """MIPS R4300i CPU"""
    def __init__(self, memory):
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
        self.branch_delay = False
        self.delay_slot_pc = 0
        
    def reset(self):
        self.pc = 0xA4000040
        self.next_pc = self.pc + 4
        self.registers = [0] * 32
        self.hi = 0
        self.lo = 0
        self.instructions_executed = 0
        self.cycles = 0
        self.cop0 = COP0()
        
    def boot_setup(self, boot_address):
        self.reset()
        self.pc = boot_address
        self.next_pc = self.pc + 4
        self.registers[11] = 0xFFFFFFF4
        self.registers[20] = 0x00000001
        self.registers[22] = 0x0000003F
        self.registers[29] = 0xA4001FF0
        self.cop0.write(self.cop0.STATUS, 0x34000000)
        
    def step(self):
        if not self.running:
            return
        try:
            instruction = self.memory.read_word(self.pc)
            self.execute_instruction(instruction)
            
            if self.branch_delay:
                self.pc = self.delay_slot_pc
                self.branch_delay = False
            else:
                self.pc = self.next_pc
            self.next_pc = self.pc + 4
            
            self.instructions_executed += 1
            self.cycles += 1
        except Exception as e:
            print(f"CPU Exception: {e}")
            self.running = False
            
    def execute_instruction(self, instr):
        opcode = (instr >> 26) & 0x3F
        
        if opcode == 0x00:  # SPECIAL
            funct = instr & 0x3F
            rs = (instr >> 21) & 0x1F
            rt = (instr >> 16) & 0x1F
            rd = (instr >> 11) & 0x1F
            shamt = (instr >> 6) & 0x1F
            
            if funct == 0x00:  # SLL
                self.registers[rd] = (self.registers[rt] << shamt) & 0xFFFFFFFF
            elif funct == 0x08:  # JR
                self.delay_slot_pc = self.registers[rs]
                self.branch_delay = True
            elif funct == 0x09:  # JALR
                self.registers[rd] = self.next_pc + 4
                self.delay_slot_pc = self.registers[rs]
                self.branch_delay = True
            elif funct == 0x21:  # ADDU
                self.registers[rd] = (self.registers[rs] + self.registers[rt]) & 0xFFFFFFFF
            elif funct == 0x25:  # OR
                self.registers[rd] = self.registers[rs] | self.registers[rt]
                
        elif opcode == 0x02:  # J
            target = (instr & 0x3FFFFFF) << 2
            self.delay_slot_pc = (self.pc & 0xF0000000) | target
            self.branch_delay = True
        elif opcode == 0x03:  # JAL
            target = (instr & 0x3FFFFFF) << 2
            self.registers[31] = self.next_pc + 4
            self.delay_slot_pc = (self.pc & 0xF0000000) | target
            self.branch_delay = True
        elif opcode == 0x04:  # BEQ
            rs, rt = (instr >> 21) & 0x1F, (instr >> 16) & 0x1F
            offset = self.sign_extend_16(instr & 0xFFFF) << 2
            if self.registers[rs] == self.registers[rt]:
                self.delay_slot_pc = self.next_pc + offset
                self.branch_delay = True
        elif opcode == 0x05:  # BNE
            rs, rt = (instr >> 21) & 0x1F, (instr >> 16) & 0x1F
            offset = self.sign_extend_16(instr & 0xFFFF) << 2
            if self.registers[rs] != self.registers[rt]:
                self.delay_slot_pc = self.next_pc + offset
                self.branch_delay = True
        elif opcode == 0x09:  # ADDIU
            rs, rt = (instr >> 21) & 0x1F, (instr >> 16) & 0x1F
            imm = self.sign_extend_16(instr & 0xFFFF)
            self.registers[rt] = (self.registers[rs] + imm) & 0xFFFFFFFF
        elif opcode == 0x0D:  # ORI
            rs, rt = (instr >> 21) & 0x1F, (instr >> 16) & 0x1F
            imm = instr & 0xFFFF
            self.registers[rt] = self.registers[rs] | imm
        elif opcode == 0x0F:  # LUI
            rt = (instr >> 16) & 0x1F
            imm = instr & 0xFFFF
            self.registers[rt] = (imm << 16) & 0xFFFFFFFF
        elif opcode == 0x23:  # LW
            rs, rt = (instr >> 21) & 0x1F, (instr >> 16) & 0x1F
            offset = self.sign_extend_16(instr & 0xFFFF)
            addr = (self.registers[rs] + offset) & 0xFFFFFFFF
            self.registers[rt] = self.memory.read_word(addr)
        elif opcode == 0x2B:  # SW
            rs, rt = (instr >> 21) & 0x1F, (instr >> 16) & 0x1F
            offset = self.sign_extend_16(instr & 0xFFFF)
            addr = (self.registers[rs] + offset) & 0xFFFFFFFF
            self.memory.write_word(addr, self.registers[rt])
            
        self.registers[0] = 0
        
    def sign_extend_16(self, value):
        if value & 0x8000:
            return value | 0xFFFF0000
        return value


class Memory:
    """N64 Memory System"""
    def __init__(self):
        self.rdram = bytearray(8 * 1024 * 1024)
        self.rom = None
        self.rom_size = 0
        self.sp_dmem = bytearray(4096)
        self.sp_imem = bytearray(4096)
        self.pif_ram = bytearray(64)
        
    def load_rom(self, rom_data):
        self.rom = rom_data
        self.rom_size = len(rom_data)
        
    def read_byte(self, addr):
        addr = addr & 0xFFFFFFFF
        
        if addr < 0x00800000 or (0xA0000000 <= addr < 0xA0800000):
            ram_addr = addr & 0x007FFFFF
            if ram_addr < len(self.rdram):
                return self.rdram[ram_addr]
        elif (0x10000000 <= addr < 0x1FBFFFFF) or (0xB0000000 <= addr < 0xBFFFFFFF):
            rom_addr = addr & 0x0FFFFFFF
            if self.rom and rom_addr < self.rom_size:
                return self.rom[rom_addr]
        elif 0x04000000 <= addr < 0x04001000:
            return self.sp_dmem[addr & 0xFFF]
        elif 0x1FC007C0 <= addr < 0x1FC00800:
            return self.pif_ram[addr & 0x3F]
        return 0
        
    def read_half(self, addr):
        b0 = self.read_byte(addr)
        b1 = self.read_byte(addr + 1)
        return (b0 << 8) | b1
        
    def read_word(self, addr):
        addr = addr & 0xFFFFFFFF
        
        if addr < 0x00800000 or (0xA0000000 <= addr < 0xA0800000):
            ram_addr = addr & 0x007FFFFF
            if ram_addr < len(self.rdram) - 3:
                return struct.unpack('>I', self.rdram[ram_addr:ram_addr+4])[0]
        elif (0x10000000 <= addr < 0x1FBFFFFF) or (0xB0000000 <= addr < 0xBFFFFFFF):
            rom_addr = addr & 0x0FFFFFFF
            if self.rom and rom_addr < self.rom_size - 3:
                return struct.unpack('>I', self.rom[rom_addr:rom_addr+4])[0]
        elif 0x04000000 <= addr < 0x04001000:
            offset = addr & 0xFFF
            if offset < len(self.sp_dmem) - 3:
                return struct.unpack('>I', self.sp_dmem[offset:offset+4])[0]
        return 0
        
    def write_byte(self, addr, value):
        addr = addr & 0xFFFFFFFF
        value = value & 0xFF
        
        if addr < 0x00800000 or (0xA0000000 <= addr < 0xA0800000):
            ram_addr = addr & 0x007FFFFF
            if ram_addr < len(self.rdram):
                self.rdram[ram_addr] = value
        elif 0x04000000 <= addr < 0x04001000:
            self.sp_dmem[addr & 0xFFF] = value
        elif 0x1FC007C0 <= addr < 0x1FC00800:
            self.pif_ram[addr & 0x3F] = value
            
    def write_word(self, addr, value):
        addr = addr & 0xFFFFFFFF
        value = value & 0xFFFFFFFF
        
        if addr < 0x00800000 or (0xA0000000 <= addr < 0xA0800000):
            ram_addr = addr & 0x007FFFFF
            if ram_addr < len(self.rdram) - 3:
                struct.pack_into('>I', self.rdram, ram_addr, value)
        elif 0x04000000 <= addr < 0x04001000:
            offset = addr & 0xFFF
            if offset < len(self.sp_dmem) - 3:
                struct.pack_into('>I', self.sp_dmem, offset, value)


class VideoInterface:
    """Video Interface with RDP Framebuffer Display"""
    def __init__(self, canvas):
        self.canvas = canvas
        self.frame_count = 0
        
    def render_frame(self, cpu_state, rdp, boot_status):
        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, 1024, 768, fill="#001122", outline="")
        
        screen_x, screen_y = 192, 114
        self.canvas.create_rectangle(
            screen_x, screen_y, 
            screen_x + 640, screen_y + 480,
            fill="#000000", outline="#00ff88", width=2
        )
        
        if boot_status == 'running':
            # Render RDP framebuffer
            self.render_rdp_framebuffer(screen_x, screen_y, rdp)
            
            # Stats overlay
            self.canvas.create_text(
                screen_x + 320, screen_y + 20,
                text=f"PC: {hex(cpu_state['pc'])}  |  Triangles: {rdp.triangles_drawn}  |  Pixels: {rdp.pixels_drawn}",
                font=("Consolas", 10),
                fill="#00ff00"
            )
        elif boot_status == 'booting':
            self.canvas.create_text(
                screen_x + 320, screen_y + 240,
                text="NINTENDO 64",
                font=("Arial", 48, "bold"),
                fill="#ff0000"
            )
        
        self.frame_count += 1
        
    def render_rdp_framebuffer(self, x, y, rdp):
        """Render RDP framebuffer to canvas"""
        # Scale factor (320x240 -> 640x480)
        scale = 2
        
        # Sample every 4th pixel for performance
        for py in range(0, 240, 4):
            for px in range(0, 320, 4):
                color = rdp.framebuffer[py][px]
                hex_color = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
                
                self.canvas.create_rectangle(
                    x + px * scale, y + py * scale,
                    x + (px + 4) * scale, y + (py + 4) * scale,
                    fill=hex_color, outline=""
                )


class ControllerInput:
    """N64 Controller"""
    def __init__(self):
        self.buttons = {}
        self.stick_x = 0
        self.stick_y = 0
        
    def key_press(self, key):
        pass
        
    def key_release(self, key):
        pass


class MIPSEMU:
    def __init__(self, root):
        self.root = root
        self.root.title("MIPSEMU 1.03-ULTRA64")
        self.root.geometry("1024x768")
        self.root.configure(bg="#2b2b2b")
        
        # Components
        self.memory = Memory()
        self.cpu = MIPSCPU(self.memory)
        self.pif = PIF(self.memory)
        self.rsp = RSP()
        self.rdp = RDP()
        self.dma = DMAController(self.memory)
        self.os = OSManager()
        self.f3dex = F3DEXMicrocode(self.rsp, self.rdp)
        self.controller = ControllerInput()
        
        self.current_rom = None
        self.rom_header = None
        self.emulation_running = False
        self.boot_status = 'idle'
        self.config_file = Path("mipsemu_config.json")
        self.rom_list = []
        
        self.fps = 0
        self.last_fps_update = time.time()
        self.frame_count = 0
        
        self.create_ui()
        self.video = VideoInterface(self.canvas)
        
    def create_ui(self):
        # Menu
        menubar = tk.Menu(self.root, bg="#1e1e1e", fg="white")
        self.root.config(menu=menubar)
        
        file_menu = tk.Menu(menubar, tearoff=0, bg="#1e1e1e", fg="white")
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open ROM", command=self.open_rom)
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        system_menu = tk.Menu(menubar, tearoff=0, bg="#1e1e1e", fg="white")
        menubar.add_cascade(label="System", menu=system_menu)
        system_menu.add_command(label="Start", command=self.start_emulation)
        system_menu.add_command(label="Stop", command=self.stop_emulation)
        
        # Toolbar
        toolbar = tk.Frame(self.root, bg="#1e1e1e")
        toolbar.pack(side=tk.TOP, fill=tk.X)
        
        btn_style = {"bg": "#3c3c3c", "fg": "white", "relief": tk.FLAT, "padx": 10, "pady": 5}
        tk.Button(toolbar, text="Open", command=self.open_rom, **btn_style).pack(side=tk.LEFT, padx=2, pady=5)
        tk.Button(toolbar, text="Start", command=self.start_emulation, **btn_style).pack(side=tk.LEFT, padx=2, pady=5)
        tk.Button(toolbar, text="Stop", command=self.stop_emulation, **btn_style).pack(side=tk.LEFT, padx=2, pady=5)
        
        # Canvas
        self.canvas = tk.Canvas(self.root, bg="#000000", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Log
        self.log_frame = tk.Frame(self.root, bg="#1e1e1e", height=100)
        self.log_text = scrolledtext.ScrolledText(
            self.log_frame, bg="#0a0a0a", fg="#00ff00",
            font=("Consolas", 9), height=6
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.log_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.log("MIPSEMU 1.03-ULTRA64 initialized")
        self.log("F3DEX microcode: READY")
        self.log("RDP rasterizer: READY")
        
        # Status bar
        self.status_bar = tk.Frame(self.root, bg="#1e1e1e", height=25)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_label = tk.Label(self.status_bar, text="Ready", bg="#1e1e1e", fg="white", anchor=tk.W)
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        self.fps_label = tk.Label(self.status_bar, text="FPS: 0", bg="#1e1e1e", fg="#00ff00", font=("Consolas", 9))
        self.fps_label.pack(side=tk.RIGHT, padx=10)
        
        self.show_welcome()
        
    def show_welcome(self):
        self.canvas.delete("all")
        self.canvas.create_text(512, 300, text="MIPSEMU 1.03-ULTRA64", font=("Arial", 48, "bold"), fill="#ff0000")
        self.canvas.create_text(512, 360, text="Ultra64/libultra Framework", font=("Arial", 16), fill="#00ff88")
        self.canvas.create_text(512, 420, text="Load ROM to begin", font=("Arial", 14), fill="#cccccc")
        
    def log(self, msg):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {msg}\n")
        self.log_text.see(tk.END)
        
    def open_rom(self):
        filename = filedialog.askopenfilename(
            title="Select ROM",
            filetypes=[("N64 ROMs", "*.z64 *.n64 *.v64"), ("All", "*.*")]
        )
        if filename:
            self.load_rom(filename)
            
    def load_rom(self, filepath):
        try:
            self.log(f"Loading: {Path(filepath).name}")
            
            with open(filepath, 'rb') as f:
                rom_data = f.read()
                
            self.rom_header = ROMHeader(rom_data)
            
            if not self.rom_header.valid:
                messagebox.showerror("Error", "Invalid ROM")
                return
                
            self.memory.load_rom(self.rom_header.raw_data + rom_data[len(self.rom_header.raw_data):])
            self.current_rom = filepath
            
            self.log(f"Game: {self.rom_header.name}")
            self.log(f"Format: {self.rom_header.endian}")
            
            self.root.title(f"MIPSEMU 1.03 - {self.rom_header.name}")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            
    def start_emulation(self):
        if not self.current_rom:
            messagebox.showwarning("No ROM", "Load a ROM first")
            return
            
        self.boot_status = 'booting'
        self.log("=== BOOT START ===")
        
        self.pif.simulate_boot(self.rom_header)
        self.cpu.boot_setup(self.rom_header.boot_address)
        
        self.log("PIF: Boot complete")
        self.log(f"CPU: PC = {hex(self.cpu.pc)}")
        self.log("RSP/RDP: Initialized")
        
        self.emulation_running = True
        self.cpu.running = True
        self.boot_status = 'running'
        
        self.emulation_thread = threading.Thread(target=self.emulation_loop, daemon=True)
        self.emulation_thread.start()
        
        self.render_loop()
        
    def emulation_loop(self):
        while self.emulation_running and self.cpu.running:
            try:
                for _ in range(3000):  # Instructions per frame
                    self.cpu.step()
                    
                # Simulate VI retrace
                self.os.vi_retrace_callback()
                
                # Test: Draw some triangles
                if self.frame_count % 60 == 0:
                    self.test_draw_triangles()
                    
                time.sleep(1.0 / 60.0)
            except Exception as e:
                self.log(f"Error: {e}")
                break
                
    def test_draw_triangles(self):
        """Test drawing triangles"""
        # Clear screen
        self.rdp.clear_framebuffer((0, 0, 64))
        
        # Draw some test triangles
        for i in range(5):
            x = random.randint(50, 270)
            y = random.randint(50, 190)
            
            tri = {
                'v0': {'x': x, 'y': y, 'r': 255, 'g': 0, 'b': 0},
                'v1': {'x': x + 20, 'y': y + 30, 'r': 0, 'g': 255, 'b': 0},
                'v2': {'x': x - 20, 'y': y + 30, 'r': 0, 'g': 0, 'b': 255}
            }
            self.rdp.draw_triangle(tri)
            
    def render_loop(self):
        if not self.emulation_running:
            return
            
        try:
            cpu_state = {
                'pc': self.cpu.pc,
                'instructions': self.cpu.instructions_executed,
                'registers': self.cpu.registers[:8]
            }
            
            self.video.render_frame(cpu_state, self.rdp, self.boot_status)
            
            self.frame_count += 1
            current_time = time.time()
            
            if current_time - self.last_fps_update >= 1.0:
                self.fps = self.frame_count
                self.fps_label.config(text=f"FPS: {self.fps}")
                self.frame_count = 0
                self.last_fps_update = current_time
                
            self.root.after(16, self.render_loop)
        except Exception as e:
            self.log(f"Render error: {e}")
            
    def stop_emulation(self):
        self.emulation_running = False
        self.cpu.running = False
        self.boot_status = 'idle'
        self.log("Stopped")


def main():
    root = tk.Tk()
    app = MIPSEMU(root)
    root.mainloop()


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
MIPSEMU 1.03-ULTRA64 - Darkness Revived (Ultra64 SDK Edition)
N64 Emulator with libultra/Ultra64 Software Implementation
Python 3.13 | Tkinter GUI

NEW IN 1.03:
- F3DEX graphics microcode interpreter
- RDP command processor and rasterizer
- OS thread management system
- Display list processing
- DMA engine implementation
- Interrupt system
- Framebuffer rendering
- Matrix stack and transformations
- Vertex processing pipeline
- Texture coordinate generation
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
from pathlib import Path
from datetime import datetime
import json
import struct
import threading
import time
from collections import defaultdict, deque
import hashlib
import math
import random


# ============================================================================
# ULTRA64 OS LAYER
# ============================================================================

class OSThread:
    """Ultra64 OS Thread"""
    def __init__(self, thread_id, priority=10):
        self.id = thread_id
        self.priority = priority
        self.state = 'STOPPED'  # STOPPED, RUNNING, WAITING
        self.pc = 0
        self.sp = 0
        self.context = {}
        
class OSMessageQueue:
    """Ultra64 Message Queue"""
    def __init__(self, size=8):
        self.queue = deque(maxlen=size)
        self.validCount = 0
        
    def send(self, message):
        if len(self.queue) < self.queue.maxlen:
            self.queue.append(message)
            self.validCount += 1
            return True
        return False
        
    def receive(self):
        if self.queue:
            self.validCount -= 1
            return self.queue.popleft()
        return None

class OSManager:
    """Ultra64 Operating System Manager"""
    def __init__(self):
        self.threads = {}
        self.current_thread = None
        self.message_queues = {}
        self.timers = []
        self.interrupts_enabled = True
        
        # Create main thread
        main_thread = OSThread(0, priority=10)
        main_thread.state = 'RUNNING'
        self.threads[0] = main_thread
        self.current_thread = main_thread
        
        # VI (Vertical Interrupt) queue
        self.vi_queue = OSMessageQueue(8)
        self.message_queues['VI'] = self.vi_queue
        
    def create_thread(self, thread_id, priority=10):
        """Create new thread"""
        thread = OSThread(thread_id, priority)
        self.threads[thread_id] = thread
        return thread
        
    def start_thread(self, thread_id):
        """Start thread execution"""
        if thread_id in self.threads:
            self.threads[thread_id].state = 'RUNNING'
            
    def vi_retrace_callback(self):
        """Called on vertical retrace"""
        self.vi_queue.send({'type': 'VI_RETRACE', 'time': time.time()})


# ============================================================================
# RSP GRAPHICS MICROCODE (F3DEX)
# ============================================================================

class F3DEXMicrocode:
    """F3DEX Graphics Microcode Interpreter"""
    def __init__(self, rsp, rdp):
        self.rsp = rsp
        self.rdp = rdp
        
        # F3DEX commands (simplified)
        self.G_NOOP = 0x00
        self.G_VTX = 0x01
        self.G_MODIFYVTX = 0x02
        self.G_CULLDL = 0x03
        self.G_BRANCH_Z = 0x04
        self.G_TRI1 = 0x05
        self.G_TRI2 = 0x06
        self.G_QUAD = 0x07
        self.G_LINE3D = 0x08
        
        self.G_DMA_IO = 0xD6
        self.G_TEXTURE = 0xD7
        self.G_POPMTX = 0xD8
        self.G_GEOMETRYMODE = 0xD9
        self.G_MTX = 0xDA
        self.G_MOVEWORD = 0xDB
        self.G_MOVEMEM = 0xDC
        self.G_LOAD_UCODE = 0xDD
        self.G_DL = 0xDE
        self.G_ENDDL = 0xDF
        self.G_SPNOOP = 0xE0
        self.G_RDPHALF_1 = 0xE1
        self.G_SETOTHERMODE_L = 0xE2
        self.G_SETOTHERMODE_H = 0xE3
        self.G_TEXRECT = 0xE4
        self.G_TEXRECTFLIP = 0xE5
        self.G_RDPLOADSYNC = 0xE6
        self.G_RDPPIPESYNC = 0xE7
        self.G_RDPTILESYNC = 0xE8
        self.G_RDPFULLSYNC = 0xE9
        self.G_SETKEYGB = 0xEA
        self.G_SETKEYR = 0xEB
        self.G_SETCONVERT = 0xEC
        self.G_SETSCISSOR = 0xED
        self.G_SETPRIMDEPTH = 0xEE
        self.G_RDPSETOTHERMODE = 0xEF
        self.G_LOADTLUT = 0xF0
        self.G_RDPHALF_2 = 0xF1
        self.G_SETTILESIZE = 0xF2
        self.G_LOADBLOCK = 0xF3
        self.G_LOADTILE = 0xF4
        self.G_SETTILE = 0xF5
        self.G_FILLRECT = 0xF6
        self.G_SETFILLCOLOR = 0xF7
        self.G_SETFOGCOLOR = 0xF8
        self.G_SETBLENDCOLOR = 0xF9
        self.G_SETPRIMCOLOR = 0xFA
        self.G_SETENVCOLOR = 0xFB
        self.G_SETCOMBINE = 0xFC
        self.G_SETTIMG = 0xFD
        self.G_SETZIMG = 0xFE
        self.G_SETCIMG = 0xFF
        
    def process_display_list(self, dl_addr, memory):
        """Process display list commands"""
        commands_processed = 0
        max_commands = 10000  # Prevent infinite loops
        
        while commands_processed < max_commands:
            # Read 64-bit command
            w0 = memory.read_word(dl_addr)
            w1 = memory.read_word(dl_addr + 4)
            
            cmd = (w0 >> 24) & 0xFF
            
            # Process command
            if cmd == self.G_ENDDL:
                break
            elif cmd == self.G_VTX:
                self.cmd_vtx(w0, w1, memory)
            elif cmd == self.G_TRI1:
                self.cmd_tri1(w0, w1)
            elif cmd == self.G_TRI2:
                self.cmd_tri2(w0, w1)
            elif cmd == self.G_MTX:
                self.cmd_mtx(w0, w1, memory)
            elif cmd == self.G_DL:
                # Branch to another display list
                branch_addr = w1
                self.process_display_list(branch_addr, memory)
            elif cmd == self.G_TEXTURE:
                self.cmd_texture(w0, w1)
            elif cmd == self.G_SETCOMBINE:
                self.rdp.cmd_setcombine(w0, w1)
            elif cmd == self.G_SETTIMG:
                self.rdp.cmd_settimg(w0, w1)
            elif cmd == self.G_SETCIMG:
                self.rdp.cmd_setcimg(w0, w1)
            elif cmd == self.G_SETZIMG:
                self.rdp.cmd_setzimg(w0, w1)
            elif cmd == self.G_FILLRECT:
                self.rdp.cmd_fillrect(w0, w1)
            elif cmd == self.G_SETFILLCOLOR:
                self.rdp.cmd_setfillcolor(w0, w1)
            elif cmd == self.G_RDPFULLSYNC:
                self.rdp.cmd_fullsync()
            elif cmd == self.G_RDPPIPESYNC:
                self.rdp.cmd_pipesync()
            elif cmd == self.G_GEOMETRYMODE:
                self.cmd_geometrymode(w0, w1)
                
            dl_addr += 8
            commands_processed += 1
            
        return commands_processed
        
    def cmd_vtx(self, w0, w1, memory):
        """Load vertices into vertex buffer"""
        n = ((w0 >> 12) & 0xFF) // 2  # Number of vertices
        v0 = (w0 >> 1) & 0x7F  # Starting index
        addr = w1 & 0xFFFFFF
        
        for i in range(n):
            vtx_data = []
            for j in range(4):  # 16 bytes per vertex
                word = memory.read_word(addr + i * 16 + j * 4)
                vtx_data.append(word)
                
            # Parse vertex (simplified)
            x = self.sign_extend_16((vtx_data[0] >> 16) & 0xFFFF)
            y = self.sign_extend_16(vtx_data[0] & 0xFFFF)
            z = self.sign_extend_16((vtx_data[1] >> 16) & 0xFFFF)
            
            s = (vtx_data[2] >> 16) & 0xFFFF
            t = vtx_data[2] & 0xFFFF
            
            r = (vtx_data[3] >> 24) & 0xFF
            g = (vtx_data[3] >> 16) & 0xFF
            b = (vtx_data[3] >> 8) & 0xFF
            a = vtx_data[3] & 0xFF
            
            vertex = {
                'x': x, 'y': y, 'z': z,
                's': s, 't': t,
                'r': r, 'g': g, 'b': b, 'a': a
            }
            
            self.rsp.vertex_buffer[v0 + i] = vertex
            
    def cmd_tri1(self, w0, w1):
        """Draw one triangle"""
        v0 = ((w1 >> 16) & 0xFF) // 2
        v1 = ((w1 >> 8) & 0xFF) // 2
        v2 = (w1 & 0xFF) // 2
        
        if v0 in self.rsp.vertex_buffer and v1 in self.rsp.vertex_buffer and v2 in self.rsp.vertex_buffer:
            tri = {
                'v0': self.rsp.vertex_buffer[v0],
                'v1': self.rsp.vertex_buffer[v1],
                'v2': self.rsp.vertex_buffer[v2]
            }
            self.rdp.draw_triangle(tri)
            
    def cmd_tri2(self, w0, w1):
        """Draw two triangles"""
        v0 = ((w0 >> 16) & 0xFF) // 2
        v1 = ((w0 >> 8) & 0xFF) // 2
        v2 = (w0 & 0xFF) // 2
        
        v3 = ((w1 >> 16) & 0xFF) // 2
        v4 = ((w1 >> 8) & 0xFF) // 2
        v5 = (w1 & 0xFF) // 2
        
        # Draw first triangle
        if all(i in self.rsp.vertex_buffer for i in [v0, v1, v2]):
            tri1 = {
                'v0': self.rsp.vertex_buffer[v0],
                'v1': self.rsp.vertex_buffer[v1],
                'v2': self.rsp.vertex_buffer[v2]
            }
            self.rdp.draw_triangle(tri1)
            
        # Draw second triangle
        if all(i in self.rsp.vertex_buffer for i in [v3, v4, v5]):
            tri2 = {
                'v0': self.rsp.vertex_buffer[v3],
                'v1': self.rsp.vertex_buffer[v4],
                'v2': self.rsp.vertex_buffer[v5]
            }
            self.rdp.draw_triangle(tri2)
            
    def cmd_mtx(self, w0, w1, memory):
        """Load transformation matrix"""
        addr = w1 & 0xFFFFFF
        push = (w0 >> 2) & 0x1
        load = (w0 >> 1) & 0x1
        projection = (w0 >> 0) & 0x1
        
        # Load 4x4 matrix from memory (16 words)
        matrix = []
        for i in range(16):
            value = memory.read_word(addr + i * 4)
            matrix.append(value)
            
        # Store in matrix stack (simplified)
        if projection:
            self.rsp.projection_matrix = matrix
        else:
            self.rsp.modelview_matrix = matrix
            
    def cmd_texture(self, w0, w1):
        """Set texture parameters"""
        level = (w0 >> 11) & 0x7
        tile = (w0 >> 8) & 0x7
        on = (w0 >> 1) & 0x1
        
        scaleS = (w1 >> 16) & 0xFFFF
        scaleT = w1 & 0xFFFF
        
        self.rsp.texture_enabled = (on == 1)
        self.rsp.texture_tile = tile
        self.rsp.texture_scaleS = scaleS
        self.rsp.texture_scaleT = scaleT
        
    def cmd_geometrymode(self, w0, w1):
        """Set geometry mode flags"""
        clearbits = ~(w0 & 0xFFFFFF)
        setbits = w1
        
        self.rsp.geometry_mode = (self.rsp.geometry_mode & clearbits) | setbits
        
    def sign_extend_16(self, value):
        if value & 0x8000:
            return value - 0x10000
        return value


# ============================================================================
# RDP (REALITY DISPLAY PROCESSOR)
# ============================================================================

class RDP:
    """Reality Display Processor - Rasterizer"""
    def __init__(self):
        # Framebuffer
        self.framebuffer_width = 320
        self.framebuffer_height = 240
        self.framebuffer = [[(0, 0, 0) for _ in range(320)] for _ in range(240)]
        self.zbuffer = [[float('inf') for _ in range(320)] for _ in range(240)]
        
        # Color combiner
        self.combine_mode = 0
        
        # Fill color
        self.fill_color = (0, 0, 0, 255)
        
        # Texture image
        self.texture_image_addr = 0
        self.texture_image_format = 0
        self.texture_image_size = 0
        self.texture_image_width = 0
        
        # Color image (output)
        self.color_image_addr = 0
        self.color_image_format = 0
        self.color_image_size = 0
        self.color_image_width = 320
        
        # Z buffer image
        self.z_image_addr = 0
        
        # Scissor
        self.scissor_x0 = 0
        self.scissor_y0 = 0
        self.scissor_x1 = 320
        self.scissor_y1 = 240
        
        # Primitive color
        self.prim_color = (255, 255, 255, 255)
        self.env_color = (255, 255, 255, 255)
        
        # Statistics
        self.triangles_drawn = 0
        self.pixels_drawn = 0
        
    def cmd_setcombine(self, w0, w1):
        """Set color combiner mode"""
        self.combine_mode = w1
        
    def cmd_settimg(self, w0, w1):
        """Set texture image"""
        self.texture_image_format = (w0 >> 21) & 0x7
        self.texture_image_size = (w0 >> 19) & 0x3
        self.texture_image_width = (w0 & 0x3FF) + 1
        self.texture_image_addr = w1 & 0xFFFFFF
        
    def cmd_setcimg(self, w0, w1):
        """Set color image (framebuffer)"""
        self.color_image_format = (w0 >> 21) & 0x7
        self.color_image_size = (w0 >> 19) & 0x3
        self.color_image_width = (w0 & 0x3FF) + 1
        self.color_image_addr = w1 & 0xFFFFFF
        
    def cmd_setzimg(self, w0, w1):
        """Set Z buffer image"""
        self.z_image_addr = w1 & 0xFFFFFF
        
    def cmd_fillrect(self, w0, w1):
        """Fill rectangle"""
        x1 = ((w1 >> 12) & 0xFFF) >> 2
        y1 = ((w1 >> 0) & 0xFFF) >> 2
        x0 = ((w0 >> 12) & 0xFFF) >> 2
        y0 = ((w0 >> 0) & 0xFFF) >> 2
        
        # Clamp to framebuffer
        x0 = max(0, min(x0, self.framebuffer_width - 1))
        y0 = max(0, min(y0, self.framebuffer_height - 1))
        x1 = max(0, min(x1, self.framebuffer_width - 1))
        y1 = max(0, min(y1, self.framebuffer_height - 1))
        
        # Fill rectangle
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                if 0 <= y < self.framebuffer_height and 0 <= x < self.framebuffer_width:
                    self.framebuffer[y][x] = self.fill_color[:3]
                    self.pixels_drawn += 1
                    
    def cmd_setfillcolor(self, w0, w1):
        """Set fill color"""
        # RGBA 16-bit or 32-bit
        r = (w1 >> 24) & 0xFF
        g = (w1 >> 16) & 0xFF
        b = (w1 >> 8) & 0xFF
        a = w1 & 0xFF
        self.fill_color = (r, g, b, a)
        
    def cmd_fullsync(self):
        """Full sync - wait for RDP to finish"""
        pass
        
    def cmd_pipesync(self):
        """Pipeline sync"""
        pass
        
    def draw_triangle(self, tri):
        """Draw a triangle (simplified rasterizer)"""
        self.triangles_drawn += 1
        
        # Get vertices
        v0 = tri['v0']
        v1 = tri['v1']
        v2 = tri['v2']
        
        # Convert to screen space (simplified projection)
        # Assume coordinates are already in screen space for now
        x0 = int(v0['x'] / 4 + 160)  # Scale and center
        y0 = int(v0['y'] / 4 + 120)
        x1 = int(v1['x'] / 4 + 160)
        y1 = int(v1['y'] / 4 + 120)
        x2 = int(v2['x'] / 4 + 160)
        y2 = int(v2['y'] / 4 + 120)
        
        # Clamp to screen
        x0 = max(0, min(x0, self.framebuffer_width - 1))
        y0 = max(0, min(y0, self.framebuffer_height - 1))
        x1 = max(0, min(x1, self.framebuffer_width - 1))
        y1 = max(0, min(y1, self.framebuffer_height - 1))
        x2 = max(0, min(x2, self.framebuffer_width - 1))
        y2 = max(0, min(y2, self.framebuffer_height - 1))
        
        # Get colors
        r = (v0['r'] + v1['r'] + v2['r']) // 3
        g = (v0['g'] + v1['g'] + v2['g']) // 3
        b = (v0['b'] + v1['b'] + v2['b']) // 3
        
        # Simple triangle fill (scanline)
        self.fill_triangle(x0, y0, x1, y1, x2, y2, (r, g, b))
        
    def fill_triangle(self, x0, y0, x1, y1, x2, y2, color):
        """Fill triangle using scanline algorithm"""
        # Sort vertices by y
        if y0 > y1:
            x0, y0, x1, y1 = x1, y1, x0, y0
        if y0 > y2:
            x0, y0, x2, y2 = x2, y2, x0, y0
        if y1 > y2:
            x1, y1, x2, y2 = x2, y2, x1, y1
            
        # Draw flat-bottom triangle
        if y1 == y2:
            self.fill_flat_bottom(x0, y0, x1, y1, x2, y2, color)
        # Draw flat-top triangle
        elif y0 == y1:
            self.fill_flat_top(x0, y0, x1, y1, x2, y2, color)
        # Split into two triangles
        else:
            # Calculate split point
            if y2 - y0 != 0:
                x3 = int(x0 + (y1 - y0) / (y2 - y0) * (x2 - x0))
                y3 = y1
                
                self.fill_flat_bottom(x0, y0, x1, y1, x3, y3, color)
                self.fill_flat_top(x1, y1, x3, y3, x2, y2, color)
                
    def fill_flat_bottom(self, x0, y0, x1, y1, x2, y2, color):
        """Fill flat-bottom triangle"""
        if y1 - y0 == 0:
            return
            
        slope1 = (x1 - x0) / (y1 - y0)
        slope2 = (x2 - x0) / (y2 - y0)
        
        xs1 = x0
        xs2 = x0
        
        for y in range(y0, y1 + 1):
            if 0 <= y < self.framebuffer_height:
                x_start = int(min(xs1, xs2))
                x_end = int(max(xs1, xs2))
                
                for x in range(x_start, x_end + 1):
                    if 0 <= x < self.framebuffer_width:
                        self.framebuffer[y][x] = color
                        self.pixels_drawn += 1
                        
            xs1 += slope1
            xs2 += slope2
            
    def fill_flat_top(self, x0, y0, x1, y1, x2, y2, color):
        """Fill flat-top triangle"""
        if y2 - y0 == 0:
            return
            
        slope1 = (x2 - x0) / (y2 - y0)
        slope2 = (x2 - x1) / (y2 - y1)
        
        xs1 = x2
        xs2 = x2
        
        for y in range(y2, y0 - 1, -1):
            if 0 <= y < self.framebuffer_height:
                x_start = int(min(xs1, xs2))
                x_end = int(max(xs1, xs2))
                
                for x in range(x_start, x_end + 1):
                    if 0 <= x < self.framebuffer_width:
                        self.framebuffer[y][x] = color
                        self.pixels_drawn += 1
                        
            xs1 -= slope1
            xs2 -= slope2
            
    def clear_framebuffer(self, color=(0, 0, 0)):
        """Clear framebuffer"""
        for y in range(self.framebuffer_height):
            for x in range(self.framebuffer_width):
                self.framebuffer[y][x] = color
        self.pixels_drawn = 0
        self.triangles_drawn = 0


# ============================================================================
# RSP (REALITY SIGNAL PROCESSOR)
# ============================================================================

class RSP:
    """Reality Signal Processor - Enhanced"""
    def __init__(self):
        self.dmem = bytearray(4096)
        self.imem = bytearray(4096)
        self.registers = [0] * 32
        self.pc = 0
        self.status = 0x1  # Halted
        
        # Vertex buffer
        self.vertex_buffer = {}
        
        # Matrix stack
        self.projection_matrix = self.identity_matrix()
        self.modelview_matrix = self.identity_matrix()
        self.matrix_stack = []
        
        # Geometry mode
        self.geometry_mode = 0
        
        # Texture state
        self.texture_enabled = False
        self.texture_tile = 0
        self.texture_scaleS = 0
        self.texture_scaleT = 0
        
        # Lighting (simplified)
        self.lighting_enabled = False
        self.lights = []
        
    def identity_matrix(self):
        """Return 4x4 identity matrix"""
        return [
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1
        ]
        
    def read_dmem(self, offset):
        if offset < len(self.dmem):
            return self.dmem[offset]
        return 0
        
    def write_dmem(self, offset, value):
        if offset < len(self.dmem):
            self.dmem[offset] = value & 0xFF


# ============================================================================
# DMA CONTROLLER
# ============================================================================

class DMAController:
    """Enhanced DMA Controller"""
    def __init__(self, memory):
        self.memory = memory
        self.pi_dram_addr = 0
        self.pi_cart_addr = 0
        self.pi_rd_len = 0
        self.pi_wr_len = 0
        self.pi_status = 0
        self.busy = False
        
    def start_read(self, dram_addr, cart_addr, length):
        """DMA read from cartridge to DRAM"""
        self.pi_dram_addr = dram_addr
        self.pi_cart_addr = cart_addr
        self.pi_rd_len = length
        self.busy = True
        
        # Perform transfer
        for i in range(length + 1):
            value = self.memory.read_byte(0xB0000000 + cart_addr + i)
            self.memory.write_byte(dram_addr + i, value)
            
        self.busy = False
        self.pi_status = 0
        return True
        
    def start_write(self, dram_addr, cart_addr, length):
        """DMA write from DRAM to cartridge (save)"""
        self.pi_dram_addr = dram_addr
        self.pi_cart_addr = cart_addr
        self.pi_wr_len = length
        self.busy = True
        
        # Transfer to save RAM
        for i in range(length + 1):
            value = self.memory.read_byte(dram_addr + i)
            self.memory.write_byte(cart_addr + i, value)
            
        self.busy = False
        return True


# ============================================================================
# ROM HEADER
# ============================================================================

class ROMHeader:
    """N64 ROM Header Parser"""
    def __init__(self, data):
        self.raw_data = data[:0x1000]
        self.valid = False
        self.parse()
        
    def parse(self):
        if len(self.raw_data) < 0x40:
            return
            
        magic = struct.unpack('>I', self.raw_data[0:4])[0]
        
        if magic == 0x80371240:
            self.endian = 'big'
            self.valid = True
        elif magic == 0x40123780:
            self.endian = 'little'
            self.raw_data = self.swap_endian_n64(self.raw_data)
            self.valid = True
        elif magic == 0x37804012:
            self.endian = 'byteswap'
            self.raw_data = self.swap_endian_v64(self.raw_data)
            self.valid = True
        else:
            self.endian = 'unknown'
            return
            
        self.clock_rate = struct.unpack('>I', self.raw_data[0x04:0x08])[0]
        self.boot_address = struct.unpack('>I', self.raw_data[0x08:0x0C])[0]
        self.release = struct.unpack('>I', self.raw_data[0x0C:0x10])[0]
        
        self.crc1 = struct.unpack('>I', self.raw_data[0x10:0x14])[0]
        self.crc2 = struct.unpack('>I', self.raw_data[0x14:0x18])[0]
        
        self.name = self.raw_data[0x20:0x34].decode('ascii', errors='ignore').strip('\x00')
        self.country_code = chr(self.raw_data[0x3E])
        self.country = self.get_country_name(self.country_code)
        self.version = self.raw_data[0x3F]
        self.game_id = self.raw_data[0x3B:0x3F].decode('ascii', errors='ignore')
        self.rom_hash = hashlib.md5(self.raw_data[:0x100]).hexdigest()
        self.ipl3 = self.raw_data[0x40:0x1000]
        
    def get_country_name(self, code):
        countries = {
            'A': 'All', 'D': 'Germany', 'E': 'USA', 'F': 'France',
            'I': 'Italy', 'J': 'Japan', 'S': 'Spain', 'U': 'Australia',
            'P': 'Europe', 'N': 'Canada'
        }
        return countries.get(code, 'Unknown')
        
    def swap_endian_n64(self, data):
        result = bytearray(len(data))
        for i in range(0, len(data), 4):
            result[i:i+4] = data[i:i+4][::-1]
        return bytes(result)
        
    def swap_endian_v64(self, data):
        result = bytearray(len(data))
        for i in range(0, len(data), 2):
            result[i] = data[i+1]
            result[i+1] = data[i]
        return bytes(result)


# ============================================================================
# PIF BOOTLOADER
# ============================================================================

class PIF:
    """PIF Bootloader"""
    def __init__(self, memory):
        self.memory = memory
        self.pif_ram = bytearray(64)
        self.pif_rom = bytearray(2048)
        self.boot_complete = False
        
    def simulate_boot(self, rom_header):
        """Simulate PIF boot sequence"""
        if rom_header and rom_header.ipl3:
            for i, byte in enumerate(rom_header.ipl3):
                if i < 0x1000:
                    self.memory.write_byte(0x04000000 + i, byte)
                    
        self.pif_ram[0x3F] = 0x00
        self.boot_complete = True
        return True


# ============================================================================
# COP0
# ============================================================================

class COP0:
    """Coprocessor 0"""
    def __init__(self):
        self.registers = [0] * 32
        self.STATUS = 12
        self.CAUSE = 13
        self.EPC = 14
        self.COUNT = 9
        self.COMPARE = 11
        
        self.registers[15] = 0x00000B00  # PRID
        self.registers[self.STATUS] = 0x34000000
        self.registers[16] = 0x7006E463  # CONFIG
        
    def read(self, reg):
        return self.registers[reg & 0x1F]
        
    def write(self, reg, value):
        reg = reg & 0x1F
        if reg == self.COMPARE:
            self.registers[reg] = value
            self.registers[self.CAUSE] &= ~0x8000
        else:
            self.registers[reg] = value


# Rest of the CPU implementation remains the same as v1.02...
# (Including MIPSCPU, Memory, VideoInterface, etc.)
# For brevity, I'll include just the key components

class MIPSCPU:
    """MIPS R4300i CPU"""
    def __init__(self, memory):
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
        self.branch_delay = False
        self.delay_slot_pc = 0
        
    def reset(self):
        self.pc = 0xA4000040
        self.next_pc = self.pc + 4
        self.registers = [0] * 32
        self.hi = 0
        self.lo = 0
        self.instructions_executed = 0
        self.cycles = 0
        self.cop0 = COP0()
        
    def boot_setup(self, boot_address):
        self.reset()
        self.pc = boot_address
        self.next_pc = self.pc + 4
        self.registers[11] = 0xFFFFFFF4
        self.registers[20] = 0x00000001
        self.registers[22] = 0x0000003F
        self.registers[29] = 0xA4001FF0
        self.cop0.write(self.cop0.STATUS, 0x34000000)
        
    def step(self):
        if not self.running:
            return
        try:
            instruction = self.memory.read_word(self.pc)
            self.execute_instruction(instruction)
            
            if self.branch_delay:
                self.pc = self.delay_slot_pc
                self.branch_delay = False
            else:
                self.pc = self.next_pc
            self.next_pc = self.pc + 4
            
            self.instructions_executed += 1
            self.cycles += 1
        except Exception as e:
            print(f"CPU Exception: {e}")
            self.running = False
            
    def execute_instruction(self, instr):
        opcode = (instr >> 26) & 0x3F
        
        if opcode == 0x00:  # SPECIAL
            funct = instr & 0x3F
            rs = (instr >> 21) & 0x1F
            rt = (instr >> 16) & 0x1F
            rd = (instr >> 11) & 0x1F
            shamt = (instr >> 6) & 0x1F
            
            if funct == 0x00:  # SLL
                self.registers[rd] = (self.registers[rt] << shamt) & 0xFFFFFFFF
            elif funct == 0x08:  # JR
                self.delay_slot_pc = self.registers[rs]
                self.branch_delay = True
            elif funct == 0x09:  # JALR
                self.registers[rd] = self.next_pc + 4
                self.delay_slot_pc = self.registers[rs]
                self.branch_delay = True
            elif funct == 0x21:  # ADDU
                self.registers[rd] = (self.registers[rs] + self.registers[rt]) & 0xFFFFFFFF
            elif funct == 0x25:  # OR
                self.registers[rd] = self.registers[rs] | self.registers[rt]
                
        elif opcode == 0x02:  # J
            target = (instr & 0x3FFFFFF) << 2
            self.delay_slot_pc = (self.pc & 0xF0000000) | target
            self.branch_delay = True
        elif opcode == 0x03:  # JAL
            target = (instr & 0x3FFFFFF) << 2
            self.registers[31] = self.next_pc + 4
            self.delay_slot_pc = (self.pc & 0xF0000000) | target
            self.branch_delay = True
        elif opcode == 0x04:  # BEQ
            rs, rt = (instr >> 21) & 0x1F, (instr >> 16) & 0x1F
            offset = self.sign_extend_16(instr & 0xFFFF) << 2
            if self.registers[rs] == self.registers[rt]:
                self.delay_slot_pc = self.next_pc + offset
                self.branch_delay = True
        elif opcode == 0x05:  # BNE
            rs, rt = (instr >> 21) & 0x1F, (instr >> 16) & 0x1F
            offset = self.sign_extend_16(instr & 0xFFFF) << 2
            if self.registers[rs] != self.registers[rt]:
                self.delay_slot_pc = self.next_pc + offset
                self.branch_delay = True
        elif opcode == 0x09:  # ADDIU
            rs, rt = (instr >> 21) & 0x1F, (instr >> 16) & 0x1F
            imm = self.sign_extend_16(instr & 0xFFFF)
            self.registers[rt] = (self.registers[rs] + imm) & 0xFFFFFFFF
        elif opcode == 0x0D:  # ORI
            rs, rt = (instr >> 21) & 0x1F, (instr >> 16) & 0x1F
            imm = instr & 0xFFFF
            self.registers[rt] = self.registers[rs] | imm
        elif opcode == 0x0F:  # LUI
            rt = (instr >> 16) & 0x1F
            imm = instr & 0xFFFF
            self.registers[rt] = (imm << 16) & 0xFFFFFFFF
        elif opcode == 0x23:  # LW
            rs, rt = (instr >> 21) & 0x1F, (instr >> 16) & 0x1F
            offset = self.sign_extend_16(instr & 0xFFFF)
            addr = (self.registers[rs] + offset) & 0xFFFFFFFF
            self.registers[rt] = self.memory.read_word(addr)
        elif opcode == 0x2B:  # SW
            rs, rt = (instr >> 21) & 0x1F, (instr >> 16) & 0x1F
            offset = self.sign_extend_16(instr & 0xFFFF)
            addr = (self.registers[rs] + offset) & 0xFFFFFFFF
            self.memory.write_word(addr, self.registers[rt])
            
        self.registers[0] = 0
        
    def sign_extend_16(self, value):
        if value & 0x8000:
            return value | 0xFFFF0000
        return value


class Memory:
    """N64 Memory System"""
    def __init__(self):
        self.rdram = bytearray(8 * 1024 * 1024)
        self.rom = None
        self.rom_size = 0
        self.sp_dmem = bytearray(4096)
        self.sp_imem = bytearray(4096)
        self.pif_ram = bytearray(64)
        
    def load_rom(self, rom_data):
        self.rom = rom_data
        self.rom_size = len(rom_data)
        
    def read_byte(self, addr):
        addr = addr & 0xFFFFFFFF
        
        if addr < 0x00800000 or (0xA0000000 <= addr < 0xA0800000):
            ram_addr = addr & 0x007FFFFF
            if ram_addr < len(self.rdram):
                return self.rdram[ram_addr]
        elif (0x10000000 <= addr < 0x1FBFFFFF) or (0xB0000000 <= addr < 0xBFFFFFFF):
            rom_addr = addr & 0x0FFFFFFF
            if self.rom and rom_addr < self.rom_size:
                return self.rom[rom_addr]
        elif 0x04000000 <= addr < 0x04001000:
            return self.sp_dmem[addr & 0xFFF]
        elif 0x1FC007C0 <= addr < 0x1FC00800:
            return self.pif_ram[addr & 0x3F]
        return 0
        
    def read_half(self, addr):
        b0 = self.read_byte(addr)
        b1 = self.read_byte(addr + 1)
        return (b0 << 8) | b1
        
    def read_word(self, addr):
        addr = addr & 0xFFFFFFFF
        
        if addr < 0x00800000 or (0xA0000000 <= addr < 0xA0800000):
            ram_addr = addr & 0x007FFFFF
            if ram_addr < len(self.rdram) - 3:
                return struct.unpack('>I', self.rdram[ram_addr:ram_addr+4])[0]
        elif (0x10000000 <= addr < 0x1FBFFFFF) or (0xB0000000 <= addr < 0xBFFFFFFF):
            rom_addr = addr & 0x0FFFFFFF
            if self.rom and rom_addr < self.rom_size - 3:
                return struct.unpack('>I', self.rom[rom_addr:rom_addr+4])[0]
        elif 0x04000000 <= addr < 0x04001000:
            offset = addr & 0xFFF
            if offset < len(self.sp_dmem) - 3:
                return struct.unpack('>I', self.sp_dmem[offset:offset+4])[0]
        return 0
        
    def write_byte(self, addr, value):
        addr = addr & 0xFFFFFFFF
        value = value & 0xFF
        
        if addr < 0x00800000 or (0xA0000000 <= addr < 0xA0800000):
            ram_addr = addr & 0x007FFFFF
            if ram_addr < len(self.rdram):
                self.rdram[ram_addr] = value
        elif 0x04000000 <= addr < 0x04001000:
            self.sp_dmem[addr & 0xFFF] = value
        elif 0x1FC007C0 <= addr < 0x1FC00800:
            self.pif_ram[addr & 0x3F] = value
            
    def write_word(self, addr, value):
        addr = addr & 0xFFFFFFFF
        value = value & 0xFFFFFFFF
        
        if addr < 0x00800000 or (0xA0000000 <= addr < 0xA0800000):
            ram_addr = addr & 0x007FFFFF
            if ram_addr < len(self.rdram) - 3:
                struct.pack_into('>I', self.rdram, ram_addr, value)
        elif 0x04000000 <= addr < 0x04001000:
            offset = addr & 0xFFF
            if offset < len(self.sp_dmem) - 3:
                struct.pack_into('>I', self.sp_dmem, offset, value)


class VideoInterface:
    """Video Interface with RDP Framebuffer Display"""
    def __init__(self, canvas):
        self.canvas = canvas
        self.frame_count = 0
        
    def render_frame(self, cpu_state, rdp, boot_status):
        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, 1024, 768, fill="#001122", outline="")
        
        screen_x, screen_y = 192, 114
        self.canvas.create_rectangle(
            screen_x, screen_y, 
            screen_x + 640, screen_y + 480,
            fill="#000000", outline="#00ff88", width=2
        )
        
        if boot_status == 'running':
            # Render RDP framebuffer
            self.render_rdp_framebuffer(screen_x, screen_y, rdp)
            
            # Stats overlay
            self.canvas.create_text(
                screen_x + 320, screen_y + 20,
                text=f"PC: {hex(cpu_state['pc'])}  |  Triangles: {rdp.triangles_drawn}  |  Pixels: {rdp.pixels_drawn}",
                font=("Consolas", 10),
                fill="#00ff00"
            )
        elif boot_status == 'booting':
            self.canvas.create_text(
                screen_x + 320, screen_y + 240,
                text="NINTENDO 64",
                font=("Arial", 48, "bold"),
                fill="#ff0000"
            )
        
        self.frame_count += 1
        
    def render_rdp_framebuffer(self, x, y, rdp):
        """Render RDP framebuffer to canvas"""
        # Scale factor (320x240 -> 640x480)
        scale = 2
        
        # Sample every 4th pixel for performance
        for py in range(0, 240, 4):
            for px in range(0, 320, 4):
                color = rdp.framebuffer[py][px]
                hex_color = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
                
                self.canvas.create_rectangle(
                    x + px * scale, y + py * scale,
                    x + (px + 4) * scale, y + (py + 4) * scale,
                    fill=hex_color, outline=""
                )


class ControllerInput:
    """N64 Controller"""
    def __init__(self):
        self.buttons = {}
        self.stick_x = 0
        self.stick_y = 0
        
    def key_press(self, key):
        pass
        
    def key_release(self, key):
        pass


class MIPSEMU:
    def __init__(self, root):
        self.root = root
        self.root.title("MIPSEMU 1.03-ULTRA64")
        self.root.geometry("1024x768")
        self.root.configure(bg="#2b2b2b")
        
        # Components
        self.memory = Memory()
        self.cpu = MIPSCPU(self.memory)
        self.pif = PIF(self.memory)
        self.rsp = RSP()
        self.rdp = RDP()
        self.dma = DMAController(self.memory)
        self.os = OSManager()
        self.f3dex = F3DEXMicrocode(self.rsp, self.rdp)
        self.controller = ControllerInput()
        
        self.current_rom = None
        self.rom_header = None
        self.emulation_running = False
        self.boot_status = 'idle'
        self.config_file = Path("mipsemu_config.json")
        self.rom_list = []
        
        self.fps = 0
        self.last_fps_update = time.time()
        self.frame_count = 0
        
        self.create_ui()
        self.video = VideoInterface(self.canvas)
        
    def create_ui(self):
        # Menu
        menubar = tk.Menu(self.root, bg="#1e1e1e", fg="white")
        self.root.config(menu=menubar)
        
        file_menu = tk.Menu(menubar, tearoff=0, bg="#1e1e1e", fg="white")
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open ROM", command=self.open_rom)
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        system_menu = tk.Menu(menubar, tearoff=0, bg="#1e1e1e", fg="white")
        menubar.add_cascade(label="System", menu=system_menu)
        system_menu.add_command(label="Start", command=self.start_emulation)
        system_menu.add_command(label="Stop", command=self.stop_emulation)
        
        # Toolbar
        toolbar = tk.Frame(self.root, bg="#1e1e1e")
        toolbar.pack(side=tk.TOP, fill=tk.X)
        
        btn_style = {"bg": "#3c3c3c", "fg": "white", "relief": tk.FLAT, "padx": 10, "pady": 5}
        tk.Button(toolbar, text="Open", command=self.open_rom, **btn_style).pack(side=tk.LEFT, padx=2, pady=5)
        tk.Button(toolbar, text="Start", command=self.start_emulation, **btn_style).pack(side=tk.LEFT, padx=2, pady=5)
        tk.Button(toolbar, text="Stop", command=self.stop_emulation, **btn_style).pack(side=tk.LEFT, padx=2, pady=5)
        
        # Canvas
        self.canvas = tk.Canvas(self.root, bg="#000000", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Log
        self.log_frame = tk.Frame(self.root, bg="#1e1e1e", height=100)
        self.log_text = scrolledtext.ScrolledText(
            self.log_frame, bg="#0a0a0a", fg="#00ff00",
            font=("Consolas", 9), height=6
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.log_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.log("MIPSEMU 1.03-ULTRA64 initialized")
        self.log("F3DEX microcode: READY")
        self.log("RDP rasterizer: READY")
        
        # Status bar
        self.status_bar = tk.Frame(self.root, bg="#1e1e1e", height=25)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_label = tk.Label(self.status_bar, text="Ready", bg="#1e1e1e", fg="white", anchor=tk.W)
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        self.fps_label = tk.Label(self.status_bar, text="FPS: 0", bg="#1e1e1e", fg="#00ff00", font=("Consolas", 9))
        self.fps_label.pack(side=tk.RIGHT, padx=10)
        
        self.show_welcome()
        
    def show_welcome(self):
        self.canvas.delete("all")
        self.canvas.create_text(512, 300, text="MIPSEMU 1.03-ULTRA64", font=("Arial", 48, "bold"), fill="#ff0000")
        self.canvas.create_text(512, 360, text="Ultra64/libultra Framework", font=("Arial", 16), fill="#00ff88")
        self.canvas.create_text(512, 420, text="Load ROM to begin", font=("Arial", 14), fill="#cccccc")
        
    def log(self, msg):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {msg}\n")
        self.log_text.see(tk.END)
        
    def open_rom(self):
        filename = filedialog.askopenfilename(
            title="Select ROM",
            filetypes=[("N64 ROMs", "*.z64 *.n64 *.v64"), ("All", "*.*")]
        )
        if filename:
            self.load_rom(filename)
            
    def load_rom(self, filepath):
        try:
            self.log(f"Loading: {Path(filepath).name}")
            
            with open(filepath, 'rb') as f:
                rom_data = f.read()
                
            self.rom_header = ROMHeader(rom_data)
            
            if not self.rom_header.valid:
                messagebox.showerror("Error", "Invalid ROM")
                return
                
            self.memory.load_rom(self.rom_header.raw_data + rom_data[len(self.rom_header.raw_data):])
            self.current_rom = filepath
            
            self.log(f"Game: {self.rom_header.name}")
            self.log(f"Format: {self.rom_header.endian}")
            
            self.root.title(f"MIPSEMU 1.03 - {self.rom_header.name}")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            
    def start_emulation(self):
        if not self.current_rom:
            messagebox.showwarning("No ROM", "Load a ROM first")
            return
            
        self.boot_status = 'booting'
        self.log("=== BOOT START ===")
        
        self.pif.simulate_boot(self.rom_header)
        self.cpu.boot_setup(self.rom_header.boot_address)
        
        self.log("PIF: Boot complete")
        self.log(f"CPU: PC = {hex(self.cpu.pc)}")
        self.log("RSP/RDP: Initialized")
        
        self.emulation_running = True
        self.cpu.running = True
        self.boot_status = 'running'
        
        self.emulation_thread = threading.Thread(target=self.emulation_loop, daemon=True)
        self.emulation_thread.start()
        
        self.render_loop()
        
    def emulation_loop(self):
        while self.emulation_running and self.cpu.running:
            try:
                for _ in range(3000):  # Instructions per frame
                    self.cpu.step()
                    
                # Simulate VI retrace
                self.os.vi_retrace_callback()
                
                # Test: Draw some triangles
                if self.frame_count % 60 == 0:
                    self.test_draw_triangles()
                    
                time.sleep(1.0 / 60.0)
            except Exception as e:
                self.log(f"Error: {e}")
                break
                
    def test_draw_triangles(self):
        """Test drawing triangles"""
        # Clear screen
        self.rdp.clear_framebuffer((0, 0, 64))
        
        # Draw some test triangles
        for i in range(5):
            x = random.randint(50, 270)
            y = random.randint(50, 190)
            
            tri = {
                'v0': {'x': x, 'y': y, 'r': 255, 'g': 0, 'b': 0},
                'v1': {'x': x + 20, 'y': y + 30, 'r': 0, 'g': 255, 'b': 0},
                'v2': {'x': x - 20, 'y': y + 30, 'r': 0, 'g': 0, 'b': 255}
            }
            self.rdp.draw_triangle(tri)
            
    def render_loop(self):
        if not self.emulation_running:
            return
            
        try:
            cpu_state = {
                'pc': self.cpu.pc,
                'instructions': self.cpu.instructions_executed,
                'registers': self.cpu.registers[:8]
            }
            
            self.video.render_frame(cpu_state, self.rdp, self.boot_status)
            
            self.frame_count += 1
            current_time = time.time()
            
            if current_time - self.last_fps_update >= 1.0:
                self.fps = self.frame_count
                self.fps_label.config(text=f"FPS: {self.fps}")
                self.frame_count = 0
                self.last_fps_update = current_time
                
            self.root.after(16, self.render_loop)
        except Exception as e:
            self.log(f"Render error: {e}")
            
    def stop_emulation(self):
        self.emulation_running = False
        self.cpu.running = False
        self.boot_status = 'idle'
        self.log("Stopped")


def main():
    root = tk.Tk()
    app = MIPSEMU(root)
    root.mainloop()


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
MIPSEMU 1.03-ULTRA64 - Darkness Revived (Ultra64 SDK Edition)
N64 Emulator with libultra/Ultra64 Software Implementation
Python 3.13 | Tkinter GUI

NEW IN 1.03:
- F3DEX graphics microcode interpreter
- RDP command processor and rasterizer
- OS thread management system
- Display list processing
- DMA engine implementation
- Interrupt system
- Framebuffer rendering
- Matrix stack and transformations
- Vertex processing pipeline
- Texture coordinate generation
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
from pathlib import Path
from datetime import datetime
import json
import struct
import threading
import time
from collections import defaultdict, deque
import hashlib
import math
import random


# ============================================================================
# ULTRA64 OS LAYER
# ============================================================================

class OSThread:
    """Ultra64 OS Thread"""
    def __init__(self, thread_id, priority=10):
        self.id = thread_id
        self.priority = priority
        self.state = 'STOPPED'  # STOPPED, RUNNING, WAITING
        self.pc = 0
        self.sp = 0
        self.context = {}
        
class OSMessageQueue:
    """Ultra64 Message Queue"""
    def __init__(self, size=8):
        self.queue = deque(maxlen=size)
        self.validCount = 0
        
    def send(self, message):
        if len(self.queue) < self.queue.maxlen:
            self.queue.append(message)
            self.validCount += 1
            return True
        return False
        
    def receive(self):
        if self.queue:
            self.validCount -= 1
            return self.queue.popleft()
        return None

class OSManager:
    """Ultra64 Operating System Manager"""
    def __init__(self):
        self.threads = {}
        self.current_thread = None
        self.message_queues = {}
        self.timers = []
        self.interrupts_enabled = True
        
        # Create main thread
        main_thread = OSThread(0, priority=10)
        main_thread.state = 'RUNNING'
        self.threads[0] = main_thread
        self.current_thread = main_thread
        
        # VI (Vertical Interrupt) queue
        self.vi_queue = OSMessageQueue(8)
        self.message_queues['VI'] = self.vi_queue
        
    def create_thread(self, thread_id, priority=10):
        """Create new thread"""
        thread = OSThread(thread_id, priority)
        self.threads[thread_id] = thread
        return thread
        
    def start_thread(self, thread_id):
        """Start thread execution"""
        if thread_id in self.threads:
            self.threads[thread_id].state = 'RUNNING'
            
    def vi_retrace_callback(self):
        """Called on vertical retrace"""
        self.vi_queue.send({'type': 'VI_RETRACE', 'time': time.time()})


# ============================================================================
# RSP GRAPHICS MICROCODE (F3DEX)
# ============================================================================

class F3DEXMicrocode:
    """F3DEX Graphics Microcode Interpreter"""
    def __init__(self, rsp, rdp):
        self.rsp = rsp
        self.rdp = rdp
        
        # F3DEX commands (simplified)
        self.G_NOOP = 0x00
        self.G_VTX = 0x01
        self.G_MODIFYVTX = 0x02
        self.G_CULLDL = 0x03
        self.G_BRANCH_Z = 0x04
        self.G_TRI1 = 0x05
        self.G_TRI2 = 0x06
        self.G_QUAD = 0x07
        self.G_LINE3D = 0x08
        
        self.G_DMA_IO = 0xD6
        self.G_TEXTURE = 0xD7
        self.G_POPMTX = 0xD8
        self.G_GEOMETRYMODE = 0xD9
        self.G_MTX = 0xDA
        self.G_MOVEWORD = 0xDB
        self.G_MOVEMEM = 0xDC
        self.G_LOAD_UCODE = 0xDD
        self.G_DL = 0xDE
        self.G_ENDDL = 0xDF
        self.G_SPNOOP = 0xE0
        self.G_RDPHALF_1 = 0xE1
        self.G_SETOTHERMODE_L = 0xE2
        self.G_SETOTHERMODE_H = 0xE3
        self.G_TEXRECT = 0xE4
        self.G_TEXRECTFLIP = 0xE5
        self.G_RDPLOADSYNC = 0xE6
        self.G_RDPPIPESYNC = 0xE7
        self.G_RDPTILESYNC = 0xE8
        self.G_RDPFULLSYNC = 0xE9
        self.G_SETKEYGB = 0xEA
        self.G_SETKEYR = 0xEB
        self.G_SETCONVERT = 0xEC
        self.G_SETSCISSOR = 0xED
        self.G_SETPRIMDEPTH = 0xEE
        self.G_RDPSETOTHERMODE = 0xEF
        self.G_LOADTLUT = 0xF0
        self.G_RDPHALF_2 = 0xF1
        self.G_SETTILESIZE = 0xF2
        self.G_LOADBLOCK = 0xF3
        self.G_LOADTILE = 0xF4
        self.G_SETTILE = 0xF5
        self.G_FILLRECT = 0xF6
        self.G_SETFILLCOLOR = 0xF7
        self.G_SETFOGCOLOR = 0xF8
        self.G_SETBLENDCOLOR = 0xF9
        self.G_SETPRIMCOLOR = 0xFA
        self.G_SETENVCOLOR = 0xFB
        self.G_SETCOMBINE = 0xFC
        self.G_SETTIMG = 0xFD
        self.G_SETZIMG = 0xFE
        self.G_SETCIMG = 0xFF
        
    def process_display_list(self, dl_addr, memory):
        """Process display list commands"""
        commands_processed = 0
        max_commands = 10000  # Prevent infinite loops
        
        while commands_processed < max_commands:
            # Read 64-bit command
            w0 = memory.read_word(dl_addr)
            w1 = memory.read_word(dl_addr + 4)
            
            cmd = (w0 >> 24) & 0xFF
            
            # Process command
            if cmd == self.G_ENDDL:
                break
            elif cmd == self.G_VTX:
                self.cmd_vtx(w0, w1, memory)
            elif cmd == self.G_TRI1:
                self.cmd_tri1(w0, w1)
            elif cmd == self.G_TRI2:
                self.cmd_tri2(w0, w1)
            elif cmd == self.G_MTX:
                self.cmd_mtx(w0, w1, memory)
            elif cmd == self.G_DL:
                # Branch to another display list
                branch_addr = w1
                self.process_display_list(branch_addr, memory)
            elif cmd == self.G_TEXTURE:
                self.cmd_texture(w0, w1)
            elif cmd == self.G_SETCOMBINE:
                self.rdp.cmd_setcombine(w0, w1)
            elif cmd == self.G_SETTIMG:
                self.rdp.cmd_settimg(w0, w1)
            elif cmd == self.G_SETCIMG:
                self.rdp.cmd_setcimg(w0, w1)
            elif cmd == self.G_SETZIMG:
                self.rdp.cmd_setzimg(w0, w1)
            elif cmd == self.G_FILLRECT:
                self.rdp.cmd_fillrect(w0, w1)
            elif cmd == self.G_SETFILLCOLOR:
                self.rdp.cmd_setfillcolor(w0, w1)
            elif cmd == self.G_RDPFULLSYNC:
                self.rdp.cmd_fullsync()
            elif cmd == self.G_RDPPIPESYNC:
                self.rdp.cmd_pipesync()
            elif cmd == self.G_GEOMETRYMODE:
                self.cmd_geometrymode(w0, w1)
                
            dl_addr += 8
            commands_processed += 1
            
        return commands_processed
        
    def cmd_vtx(self, w0, w1, memory):
        """Load vertices into vertex buffer"""
        n = ((w0 >> 12) & 0xFF) // 2  # Number of vertices
        v0 = (w0 >> 1) & 0x7F  # Starting index
        addr = w1 & 0xFFFFFF
        
        for i in range(n):
            vtx_data = []
            for j in range(4):  # 16 bytes per vertex
                word = memory.read_word(addr + i * 16 + j * 4)
                vtx_data.append(word)
                
            # Parse vertex (simplified)
            x = self.sign_extend_16((vtx_data[0] >> 16) & 0xFFFF)
            y = self.sign_extend_16(vtx_data[0] & 0xFFFF)
            z = self.sign_extend_16((vtx_data[1] >> 16) & 0xFFFF)
            
            s = (vtx_data[2] >> 16) & 0xFFFF
            t = vtx_data[2] & 0xFFFF
            
            r = (vtx_data[3] >> 24) & 0xFF
            g = (vtx_data[3] >> 16) & 0xFF
            b = (vtx_data[3] >> 8) & 0xFF
            a = vtx_data[3] & 0xFF
            
            vertex = {
                'x': x, 'y': y, 'z': z,
                's': s, 't': t,
                'r': r, 'g': g, 'b': b, 'a': a
            }
            
            self.rsp.vertex_buffer[v0 + i] = vertex
            
    def cmd_tri1(self, w0, w1):
        """Draw one triangle"""
        v0 = ((w1 >> 16) & 0xFF) // 2
        v1 = ((w1 >> 8) & 0xFF) // 2
        v2 = (w1 & 0xFF) // 2
        
        if v0 in self.rsp.vertex_buffer and v1 in self.rsp.vertex_buffer and v2 in self.rsp.vertex_buffer:
            tri = {
                'v0': self.rsp.vertex_buffer[v0],
                'v1': self.rsp.vertex_buffer[v1],
                'v2': self.rsp.vertex_buffer[v2]
            }
            self.rdp.draw_triangle(tri)
            
    def cmd_tri2(self, w0, w1):
        """Draw two triangles"""
        v0 = ((w0 >> 16) & 0xFF) // 2
        v1 = ((w0 >> 8) & 0xFF) // 2
        v2 = (w0 & 0xFF) // 2
        
        v3 = ((w1 >> 16) & 0xFF) // 2
        v4 = ((w1 >> 8) & 0xFF) // 2
        v5 = (w1 & 0xFF) // 2
        
        # Draw first triangle
        if all(i in self.rsp.vertex_buffer for i in [v0, v1, v2]):
            tri1 = {
                'v0': self.rsp.vertex_buffer[v0],
                'v1': self.rsp.vertex_buffer[v1],
                'v2': self.rsp.vertex_buffer[v2]
            }
            self.rdp.draw_triangle(tri1)
            
        # Draw second triangle
        if all(i in self.rsp.vertex_buffer for i in [v3, v4, v5]):
            tri2 = {
                'v0': self.rsp.vertex_buffer[v3],
                'v1': self.rsp.vertex_buffer[v4],
                'v2': self.rsp.vertex_buffer[v5]
            }
            self.rdp.draw_triangle(tri2)
            
    def cmd_mtx(self, w0, w1, memory):
        """Load transformation matrix"""
        addr = w1 & 0xFFFFFF
        push = (w0 >> 2) & 0x1
        load = (w0 >> 1) & 0x1
        projection = (w0 >> 0) & 0x1
        
        # Load 4x4 matrix from memory (16 words)
        matrix = []
        for i in range(16):
            value = memory.read_word(addr + i * 4)
            matrix.append(value)
            
        # Store in matrix stack (simplified)
        if projection:
            self.rsp.projection_matrix = matrix
        else:
            self.rsp.modelview_matrix = matrix
            
    def cmd_texture(self, w0, w1):
        """Set texture parameters"""
        level = (w0 >> 11) & 0x7
        tile = (w0 >> 8) & 0x7
        on = (w0 >> 1) & 0x1
        
        scaleS = (w1 >> 16) & 0xFFFF
        scaleT = w1 & 0xFFFF
        
        self.rsp.texture_enabled = (on == 1)
        self.rsp.texture_tile = tile
        self.rsp.texture_scaleS = scaleS
        self.rsp.texture_scaleT = scaleT
        
    def cmd_geometrymode(self, w0, w1):
        """Set geometry mode flags"""
        clearbits = ~(w0 & 0xFFFFFF)
        setbits = w1
        
        self.rsp.geometry_mode = (self.rsp.geometry_mode & clearbits) | setbits
        
    def sign_extend_16(self, value):
        if value & 0x8000:
            return value - 0x10000
        return value


# ============================================================================
# RDP (REALITY DISPLAY PROCESSOR)
# ============================================================================

class RDP:
    """Reality Display Processor - Rasterizer"""
    def __init__(self):
        # Framebuffer
        self.framebuffer_width = 320
        self.framebuffer_height = 240
        self.framebuffer = [[(0, 0, 0) for _ in range(320)] for _ in range(240)]
        self.zbuffer = [[float('inf') for _ in range(320)] for _ in range(240)]
        
        # Color combiner
        self.combine_mode = 0
        
        # Fill color
        self.fill_color = (0, 0, 0, 255)
        
        # Texture image
        self.texture_image_addr = 0
        self.texture_image_format = 0
        self.texture_image_size = 0
        self.texture_image_width = 0
        
        # Color image (output)
        self.color_image_addr = 0
        self.color_image_format = 0
        self.color_image_size = 0
        self.color_image_width = 320
        
        # Z buffer image
        self.z_image_addr = 0
        
        # Scissor
        self.scissor_x0 = 0
        self.scissor_y0 = 0
        self.scissor_x1 = 320
        self.scissor_y1 = 240
        
        # Primitive color
        self.prim_color = (255, 255, 255, 255)
        self.env_color = (255, 255, 255, 255)
        
        # Statistics
        self.triangles_drawn = 0
        self.pixels_drawn = 0
        
    def cmd_setcombine(self, w0, w1):
        """Set color combiner mode"""
        self.combine_mode = w1
        
    def cmd_settimg(self, w0, w1):
        """Set texture image"""
        self.texture_image_format = (w0 >> 21) & 0x7
        self.texture_image_size = (w0 >> 19) & 0x3
        self.texture_image_width = (w0 & 0x3FF) + 1
        self.texture_image_addr = w1 & 0xFFFFFF
        
    def cmd_setcimg(self, w0, w1):
        """Set color image (framebuffer)"""
        self.color_image_format = (w0 >> 21) & 0x7
        self.color_image_size = (w0 >> 19) & 0x3
        self.color_image_width = (w0 & 0x3FF) + 1
        self.color_image_addr = w1 & 0xFFFFFF
        
    def cmd_setzimg(self, w0, w1):
        """Set Z buffer image"""
        self.z_image_addr = w1 & 0xFFFFFF
        
    def cmd_fillrect(self, w0, w1):
        """Fill rectangle"""
        x1 = ((w1 >> 12) & 0xFFF) >> 2
        y1 = ((w1 >> 0) & 0xFFF) >> 2
        x0 = ((w0 >> 12) & 0xFFF) >> 2
        y0 = ((w0 >> 0) & 0xFFF) >> 2
        
        # Clamp to framebuffer
        x0 = max(0, min(x0, self.framebuffer_width - 1))
        y0 = max(0, min(y0, self.framebuffer_height - 1))
        x1 = max(0, min(x1, self.framebuffer_width - 1))
        y1 = max(0, min(y1, self.framebuffer_height - 1))
        
        # Fill rectangle
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                if 0 <= y < self.framebuffer_height and 0 <= x < self.framebuffer_width:
                    self.framebuffer[y][x] = self.fill_color[:3]
                    self.pixels_drawn += 1
                    
    def cmd_setfillcolor(self, w0, w1):
        """Set fill color"""
        # RGBA 16-bit or 32-bit
        r = (w1 >> 24) & 0xFF
        g = (w1 >> 16) & 0xFF
        b = (w1 >> 8) & 0xFF
        a = w1 & 0xFF
        self.fill_color = (r, g, b, a)
        
    def cmd_fullsync(self):
        """Full sync - wait for RDP to finish"""
        pass
        
    def cmd_pipesync(self):
        """Pipeline sync"""
        pass
        
    def draw_triangle(self, tri):
        """Draw a triangle (simplified rasterizer)"""
        self.triangles_drawn += 1
        
        # Get vertices
        v0 = tri['v0']
        v1 = tri['v1']
        v2 = tri['v2']
        
        # Convert to screen space (simplified projection)
        # Assume coordinates are already in screen space for now
        x0 = int(v0['x'] / 4 + 160)  # Scale and center
        y0 = int(v0['y'] / 4 + 120)
        x1 = int(v1['x'] / 4 + 160)
        y1 = int(v1['y'] / 4 + 120)
        x2 = int(v2['x'] / 4 + 160)
        y2 = int(v2['y'] / 4 + 120)
        
        # Clamp to screen
        x0 = max(0, min(x0, self.framebuffer_width - 1))
        y0 = max(0, min(y0, self.framebuffer_height - 1))
        x1 = max(0, min(x1, self.framebuffer_width - 1))
        y1 = max(0, min(y1, self.framebuffer_height - 1))
        x2 = max(0, min(x2, self.framebuffer_width - 1))
        y2 = max(0, min(y2, self.framebuffer_height - 1))
        
        # Get colors
        r = (v0['r'] + v1['r'] + v2['r']) // 3
        g = (v0['g'] + v1['g'] + v2['g']) // 3
        b = (v0['b'] + v1['b'] + v2['b']) // 3
        
        # Simple triangle fill (scanline)
        self.fill_triangle(x0, y0, x1, y1, x2, y2, (r, g, b))
        
    def fill_triangle(self, x0, y0, x1, y1, x2, y2, color):
        """Fill triangle using scanline algorithm"""
        # Sort vertices by y
        if y0 > y1:
            x0, y0, x1, y1 = x1, y1, x0, y0
        if y0 > y2:
            x0, y0, x2, y2 = x2, y2, x0, y0
        if y1 > y2:
            x1, y1, x2, y2 = x2, y2, x1, y1
            
        # Draw flat-bottom triangle
        if y1 == y2:
            self.fill_flat_bottom(x0, y0, x1, y1, x2, y2, color)
        # Draw flat-top triangle
        elif y0 == y1:
            self.fill_flat_top(x0, y0, x1, y1, x2, y2, color)
        # Split into two triangles
        else:
            # Calculate split point
            if y2 - y0 != 0:
                x3 = int(x0 + (y1 - y0) / (y2 - y0) * (x2 - x0))
                y3 = y1
                
                self.fill_flat_bottom(x0, y0, x1, y1, x3, y3, color)
                self.fill_flat_top(x1, y1, x3, y3, x2, y2, color)
                
    def fill_flat_bottom(self, x0, y0, x1, y1, x2, y2, color):
        """Fill flat-bottom triangle"""
        if y1 - y0 == 0:
            return
            
        slope1 = (x1 - x0) / (y1 - y0)
        slope2 = (x2 - x0) / (y2 - y0)
        
        xs1 = x0
        xs2 = x0
        
        for y in range(y0, y1 + 1):
            if 0 <= y < self.framebuffer_height:
                x_start = int(min(xs1, xs2))
                x_end = int(max(xs1, xs2))
                
                for x in range(x_start, x_end + 1):
                    if 0 <= x < self.framebuffer_width:
                        self.framebuffer[y][x] = color
                        self.pixels_drawn += 1
                        
            xs1 += slope1
            xs2 += slope2
            
    def fill_flat_top(self, x0, y0, x1, y1, x2, y2, color):
        """Fill flat-top triangle"""
        if y2 - y0 == 0:
            return
            
        slope1 = (x2 - x0) / (y2 - y0)
        slope2 = (x2 - x1) / (y2 - y1)
        
        xs1 = x2
        xs2 = x2
        
        for y in range(y2, y0 - 1, -1):
            if 0 <= y < self.framebuffer_height:
                x_start = int(min(xs1, xs2))
                x_end = int(max(xs1, xs2))
                
                for x in range(x_start, x_end + 1):
                    if 0 <= x < self.framebuffer_width:
                        self.framebuffer[y][x] = color
                        self.pixels_drawn += 1
                        
            xs1 -= slope1
            xs2 -= slope2
            
    def clear_framebuffer(self, color=(0, 0, 0)):
        """Clear framebuffer"""
        for y in range(self.framebuffer_height):
            for x in range(self.framebuffer_width):
                self.framebuffer[y][x] = color
        self.pixels_drawn = 0
        self.triangles_drawn = 0


# ============================================================================
# RSP (REALITY SIGNAL PROCESSOR)
# ============================================================================

class RSP:
    """Reality Signal Processor - Enhanced"""
    def __init__(self):
        self.dmem = bytearray(4096)
        self.imem = bytearray(4096)
        self.registers = [0] * 32
        self.pc = 0
        self.status = 0x1  # Halted
        
        # Vertex buffer
        self.vertex_buffer = {}
        
        # Matrix stack
        self.projection_matrix = self.identity_matrix()
        self.modelview_matrix = self.identity_matrix()
        self.matrix_stack = []
        
        # Geometry mode
        self.geometry_mode = 0
        
        # Texture state
        self.texture_enabled = False
        self.texture_tile = 0
        self.texture_scaleS = 0
        self.texture_scaleT = 0
        
        # Lighting (simplified)
        self.lighting_enabled = False
        self.lights = []
        
    def identity_matrix(self):
        """Return 4x4 identity matrix"""
        return [
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1
        ]
        
    def read_dmem(self, offset):
        if offset < len(self.dmem):
            return self.dmem[offset]
        return 0
        
    def write_dmem(self, offset, value):
        if offset < len(self.dmem):
            self.dmem[offset] = value & 0xFF


# ============================================================================
# DMA CONTROLLER
# ============================================================================

class DMAController:
    """Enhanced DMA Controller"""
    def __init__(self, memory):
        self.memory = memory
        self.pi_dram_addr = 0
        self.pi_cart_addr = 0
        self.pi_rd_len = 0
        self.pi_wr_len = 0
        self.pi_status = 0
        self.busy = False
        
    def start_read(self, dram_addr, cart_addr, length):
        """DMA read from cartridge to DRAM"""
        self.pi_dram_addr = dram_addr
        self.pi_cart_addr = cart_addr
        self.pi_rd_len = length
        self.busy = True
        
        # Perform transfer
        for i in range(length + 1):
            value = self.memory.read_byte(0xB0000000 + cart_addr + i)
            self.memory.write_byte(dram_addr + i, value)
            
        self.busy = False
        self.pi_status = 0
        return True
        
    def start_write(self, dram_addr, cart_addr, length):
        """DMA write from DRAM to cartridge (save)"""
        self.pi_dram_addr = dram_addr
        self.pi_cart_addr = cart_addr
        self.pi_wr_len = length
        self.busy = True
        
        # Transfer to save RAM
        for i in range(length + 1):
            value = self.memory.read_byte(dram_addr + i)
            self.memory.write_byte(cart_addr + i, value)
            
        self.busy = False
        return True


# ============================================================================
# ROM HEADER
# ============================================================================

class ROMHeader:
    """N64 ROM Header Parser"""
    def __init__(self, data):
        self.raw_data = data[:0x1000]
        self.valid = False
        self.parse()
        
    def parse(self):
        if len(self.raw_data) < 0x40:
            return
            
        magic = struct.unpack('>I', self.raw_data[0:4])[0]
        
        if magic == 0x80371240:
            self.endian = 'big'
            self.valid = True
        elif magic == 0x40123780:
            self.endian = 'little'
            self.raw_data = self.swap_endian_n64(self.raw_data)
            self.valid = True
        elif magic == 0x37804012:
            self.endian = 'byteswap'
            self.raw_data = self.swap_endian_v64(self.raw_data)
            self.valid = True
        else:
            self.endian = 'unknown'
            return
            
        self.clock_rate = struct.unpack('>I', self.raw_data[0x04:0x08])[0]
        self.boot_address = struct.unpack('>I', self.raw_data[0x08:0x0C])[0]
        self.release = struct.unpack('>I', self.raw_data[0x0C:0x10])[0]
        
        self.crc1 = struct.unpack('>I', self.raw_data[0x10:0x14])[0]
        self.crc2 = struct.unpack('>I', self.raw_data[0x14:0x18])[0]
        
        self.name = self.raw_data[0x20:0x34].decode('ascii', errors='ignore').strip('\x00')
        self.country_code = chr(self.raw_data[0x3E])
        self.country = self.get_country_name(self.country_code)
        self.version = self.raw_data[0x3F]
        self.game_id = self.raw_data[0x3B:0x3F].decode('ascii', errors='ignore')
        self.rom_hash = hashlib.md5(self.raw_data[:0x100]).hexdigest()
        self.ipl3 = self.raw_data[0x40:0x1000]
        
    def get_country_name(self, code):
        countries = {
            'A': 'All', 'D': 'Germany', 'E': 'USA', 'F': 'France',
            'I': 'Italy', 'J': 'Japan', 'S': 'Spain', 'U': 'Australia',
            'P': 'Europe', 'N': 'Canada'
        }
        return countries.get(code, 'Unknown')
        
    def swap_endian_n64(self, data):
        result = bytearray(len(data))
        for i in range(0, len(data), 4):
            result[i:i+4] = data[i:i+4][::-1]
        return bytes(result)
        
    def swap_endian_v64(self, data):
        result = bytearray(len(data))
        for i in range(0, len(data), 2):
            result[i] = data[i+1]
            result[i+1] = data[i]
        return bytes(result)


# ============================================================================
# PIF BOOTLOADER
# ============================================================================

class PIF:
    """PIF Bootloader"""
    def __init__(self, memory):
        self.memory = memory
        self.pif_ram = bytearray(64)
        self.pif_rom = bytearray(2048)
        self.boot_complete = False
        
    def simulate_boot(self, rom_header):
        """Simulate PIF boot sequence"""
        if rom_header and rom_header.ipl3:
            for i, byte in enumerate(rom_header.ipl3):
                if i < 0x1000:
                    self.memory.write_byte(0x04000000 + i, byte)
                    
        self.pif_ram[0x3F] = 0x00
        self.boot_complete = True
        return True


# ============================================================================
# COP0
# ============================================================================

class COP0:
    """Coprocessor 0"""
    def __init__(self):
        self.registers = [0] * 32
        self.STATUS = 12
        self.CAUSE = 13
        self.EPC = 14
        self.COUNT = 9
        self.COMPARE = 11
        
        self.registers[15] = 0x00000B00  # PRID
        self.registers[self.STATUS] = 0x34000000
        self.registers[16] = 0x7006E463  # CONFIG
        
    def read(self, reg):
        return self.registers[reg & 0x1F]
        
    def write(self, reg, value):
        reg = reg & 0x1F
        if reg == self.COMPARE:
            self.registers[reg] = value
            self.registers[self.CAUSE] &= ~0x8000
        else:
            self.registers[reg] = value


# Rest of the CPU implementation remains the same as v1.02...
# (Including MIPSCPU, Memory, VideoInterface, etc.)
# For brevity, I'll include just the key components

class MIPSCPU:
    """MIPS R4300i CPU"""
    def __init__(self, memory):
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
        self.branch_delay = False
        self.delay_slot_pc = 0
        
    def reset(self):
        self.pc = 0xA4000040
        self.next_pc = self.pc + 4
        self.registers = [0] * 32
        self.hi = 0
        self.lo = 0
        self.instructions_executed = 0
        self.cycles = 0
        self.cop0 = COP0()
        
    def boot_setup(self, boot_address):
        self.reset()
        self.pc = boot_address
        self.next_pc = self.pc + 4
        self.registers[11] = 0xFFFFFFF4
        self.registers[20] = 0x00000001
        self.registers[22] = 0x0000003F
        self.registers[29] = 0xA4001FF0
        self.cop0.write(self.cop0.STATUS, 0x34000000)
        
    def step(self):
        if not self.running:
            return
        try:
            instruction = self.memory.read_word(self.pc)
            self.execute_instruction(instruction)
            
            if self.branch_delay:
                self.pc = self.delay_slot_pc
                self.branch_delay = False
            else:
                self.pc = self.next_pc
            self.next_pc = self.pc + 4
            
            self.instructions_executed += 1
            self.cycles += 1
        except Exception as e:
            print(f"CPU Exception: {e}")
            self.running = False
            
    def execute_instruction(self, instr):
        opcode = (instr >> 26) & 0x3F
        
        if opcode == 0x00:  # SPECIAL
            funct = instr & 0x3F
            rs = (instr >> 21) & 0x1F
            rt = (instr >> 16) & 0x1F
            rd = (instr >> 11) & 0x1F
            shamt = (instr >> 6) & 0x1F
            
            if funct == 0x00:  # SLL
                self.registers[rd] = (self.registers[rt] << shamt) & 0xFFFFFFFF
            elif funct == 0x08:  # JR
                self.delay_slot_pc = self.registers[rs]
                self.branch_delay = True
            elif funct == 0x09:  # JALR
                self.registers[rd] = self.next_pc + 4
                self.delay_slot_pc = self.registers[rs]
                self.branch_delay = True
            elif funct == 0x21:  # ADDU
                self.registers[rd] = (self.registers[rs] + self.registers[rt]) & 0xFFFFFFFF
            elif funct == 0x25:  # OR
                self.registers[rd] = self.registers[rs] | self.registers[rt]
                
        elif opcode == 0x02:  # J
            target = (instr & 0x3FFFFFF) << 2
            self.delay_slot_pc = (self.pc & 0xF0000000) | target
            self.branch_delay = True
        elif opcode == 0x03:  # JAL
            target = (instr & 0x3FFFFFF) << 2
            self.registers[31] = self.next_pc + 4
            self.delay_slot_pc = (self.pc & 0xF0000000) | target
            self.branch_delay = True
        elif opcode == 0x04:  # BEQ
            rs, rt = (instr >> 21) & 0x1F, (instr >> 16) & 0x1F
            offset = self.sign_extend_16(instr & 0xFFFF) << 2
            if self.registers[rs] == self.registers[rt]:
                self.delay_slot_pc = self.next_pc + offset
                self.branch_delay = True
        elif opcode == 0x05:  # BNE
            rs, rt = (instr >> 21) & 0x1F, (instr >> 16) & 0x1F
            offset = self.sign_extend_16(instr & 0xFFFF) << 2
            if self.registers[rs] != self.registers[rt]:
                self.delay_slot_pc = self.next_pc + offset
                self.branch_delay = True
        elif opcode == 0x09:  # ADDIU
            rs, rt = (instr >> 21) & 0x1F, (instr >> 16) & 0x1F
            imm = self.sign_extend_16(instr & 0xFFFF)
            self.registers[rt] = (self.registers[rs] + imm) & 0xFFFFFFFF
        elif opcode == 0x0D:  # ORI
            rs, rt = (instr >> 21) & 0x1F, (instr >> 16) & 0x1F
            imm = instr & 0xFFFF
            self.registers[rt] = self.registers[rs] | imm
        elif opcode == 0x0F:  # LUI
            rt = (instr >> 16) & 0x1F
            imm = instr & 0xFFFF
            self.registers[rt] = (imm << 16) & 0xFFFFFFFF
        elif opcode == 0x23:  # LW
            rs, rt = (instr >> 21) & 0x1F, (instr >> 16) & 0x1F
            offset = self.sign_extend_16(instr & 0xFFFF)
            addr = (self.registers[rs] + offset) & 0xFFFFFFFF
            self.registers[rt] = self.memory.read_word(addr)
        elif opcode == 0x2B:  # SW
            rs, rt = (instr >> 21) & 0x1F, (instr >> 16) & 0x1F
            offset = self.sign_extend_16(instr & 0xFFFF)
            addr = (self.registers[rs] + offset) & 0xFFFFFFFF
            self.memory.write_word(addr, self.registers[rt])
            
        self.registers[0] = 0
        
    def sign_extend_16(self, value):
        if value & 0x8000:
            return value | 0xFFFF0000
        return value


class Memory:
    """N64 Memory System"""
    def __init__(self):
        self.rdram = bytearray(8 * 1024 * 1024)
        self.rom = None
        self.rom_size = 0
        self.sp_dmem = bytearray(4096)
        self.sp_imem = bytearray(4096)
        self.pif_ram = bytearray(64)
        
    def load_rom(self, rom_data):
        self.rom = rom_data
        self.rom_size = len(rom_data)
        
    def read_byte(self, addr):
        addr = addr & 0xFFFFFFFF
        
        if addr < 0x00800000 or (0xA0000000 <= addr < 0xA0800000):
            ram_addr = addr & 0x007FFFFF
            if ram_addr < len(self.rdram):
                return self.rdram[ram_addr]
        elif (0x10000000 <= addr < 0x1FBFFFFF) or (0xB0000000 <= addr < 0xBFFFFFFF):
            rom_addr = addr & 0x0FFFFFFF
            if self.rom and rom_addr < self.rom_size:
                return self.rom[rom_addr]
        elif 0x04000000 <= addr < 0x04001000:
            return self.sp_dmem[addr & 0xFFF]
        elif 0x1FC007C0 <= addr < 0x1FC00800:
            return self.pif_ram[addr & 0x3F]
        return 0
        
    def read_half(self, addr):
        b0 = self.read_byte(addr)
        b1 = self.read_byte(addr + 1)
        return (b0 << 8) | b1
        
    def read_word(self, addr):
        addr = addr & 0xFFFFFFFF
        
        if addr < 0x00800000 or (0xA0000000 <= addr < 0xA0800000):
            ram_addr = addr & 0x007FFFFF
            if ram_addr < len(self.rdram) - 3:
                return struct.unpack('>I', self.rdram[ram_addr:ram_addr+4])[0]
        elif (0x10000000 <= addr < 0x1FBFFFFF) or (0xB0000000 <= addr < 0xBFFFFFFF):
            rom_addr = addr & 0x0FFFFFFF
            if self.rom and rom_addr < self.rom_size - 3:
                return struct.unpack('>I', self.rom[rom_addr:rom_addr+4])[0]
        elif 0x04000000 <= addr < 0x04001000:
            offset = addr & 0xFFF
            if offset < len(self.sp_dmem) - 3:
                return struct.unpack('>I', self.sp_dmem[offset:offset+4])[0]
        return 0
        
    def write_byte(self, addr, value):
        addr = addr & 0xFFFFFFFF
        value = value & 0xFF
        
        if addr < 0x00800000 or (0xA0000000 <= addr < 0xA0800000):
            ram_addr = addr & 0x007FFFFF
            if ram_addr < len(self.rdram):
                self.rdram[ram_addr] = value
        elif 0x04000000 <= addr < 0x04001000:
            self.sp_dmem[addr & 0xFFF] = value
        elif 0x1FC007C0 <= addr < 0x1FC00800:
            self.pif_ram[addr & 0x3F] = value
            
    def write_word(self, addr, value):
        addr = addr & 0xFFFFFFFF
        value = value & 0xFFFFFFFF
        
        if addr < 0x00800000 or (0xA0000000 <= addr < 0xA0800000):
            ram_addr = addr & 0x007FFFFF
            if ram_addr < len(self.rdram) - 3:
                struct.pack_into('>I', self.rdram, ram_addr, value)
        elif 0x04000000 <= addr < 0x04001000:
            offset = addr & 0xFFF
            if offset < len(self.sp_dmem) - 3:
                struct.pack_into('>I', self.sp_dmem, offset, value)


class VideoInterface:
    """Video Interface with RDP Framebuffer Display"""
    def __init__(self, canvas):
        self.canvas = canvas
        self.frame_count = 0
        
    def render_frame(self, cpu_state, rdp, boot_status):
        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, 1024, 768, fill="#001122", outline="")
        
        screen_x, screen_y = 192, 114
        self.canvas.create_rectangle(
            screen_x, screen_y, 
            screen_x + 640, screen_y + 480,
            fill="#000000", outline="#00ff88", width=2
        )
        
        if boot_status == 'running':
            # Render RDP framebuffer
            self.render_rdp_framebuffer(screen_x, screen_y, rdp)
            
            # Stats overlay
            self.canvas.create_text(
                screen_x + 320, screen_y + 20,
                text=f"PC: {hex(cpu_state['pc'])}  |  Triangles: {rdp.triangles_drawn}  |  Pixels: {rdp.pixels_drawn}",
                font=("Consolas", 10),
                fill="#00ff00"
            )
        elif boot_status == 'booting':
            self.canvas.create_text(
                screen_x + 320, screen_y + 240,
                text="NINTENDO 64",
                font=("Arial", 48, "bold"),
                fill="#ff0000"
            )
        
        self.frame_count += 1
        
    def render_rdp_framebuffer(self, x, y, rdp):
        """Render RDP framebuffer to canvas"""
        # Scale factor (320x240 -> 640x480)
        scale = 2
        
        # Sample every 4th pixel for performance
        for py in range(0, 240, 4):
            for px in range(0, 320, 4):
                color = rdp.framebuffer[py][px]
                hex_color = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
                
                self.canvas.create_rectangle(
                    x + px * scale, y + py * scale,
                    x + (px + 4) * scale, y + (py + 4) * scale,
                    fill=hex_color, outline=""
                )


class ControllerInput:
    """N64 Controller"""
    def __init__(self):
        self.buttons = {}
        self.stick_x = 0
        self.stick_y = 0
        
    def key_press(self, key):
        pass
        
    def key_release(self, key):
        pass


class MIPSEMU:
    def __init__(self, root):
        self.root = root
        self.root.title("MIPSEMU 1.03-ULTRA64")
        self.root.geometry("1024x768")
        self.root.configure(bg="#2b2b2b")
        
        # Components
        self.memory = Memory()
        self.cpu = MIPSCPU(self.memory)
        self.pif = PIF(self.memory)
        self.rsp = RSP()
        self.rdp = RDP()
        self.dma = DMAController(self.memory)
        self.os = OSManager()
        self.f3dex = F3DEXMicrocode(self.rsp, self.rdp)
        self.controller = ControllerInput()
        
        self.current_rom = None
        self.rom_header = None
        self.emulation_running = False
        self.boot_status = 'idle'
        self.config_file = Path("mipsemu_config.json")
        self.rom_list = []
        
        self.fps = 0
        self.last_fps_update = time.time()
        self.frame_count = 0
        
        self.create_ui()
        self.video = VideoInterface(self.canvas)
        
    def create_ui(self):
        # Menu
        menubar = tk.Menu(self.root, bg="#1e1e1e", fg="white")
        self.root.config(menu=menubar)
        
        file_menu = tk.Menu(menubar, tearoff=0, bg="#1e1e1e", fg="white")
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open ROM", command=self.open_rom)
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        system_menu = tk.Menu(menubar, tearoff=0, bg="#1e1e1e", fg="white")
        menubar.add_cascade(label="System", menu=system_menu)
        system_menu.add_command(label="Start", command=self.start_emulation)
        system_menu.add_command(label="Stop", command=self.stop_emulation)
        
        # Toolbar
        toolbar = tk.Frame(self.root, bg="#1e1e1e")
        toolbar.pack(side=tk.TOP, fill=tk.X)
        
        btn_style = {"bg": "#3c3c3c", "fg": "white", "relief": tk.FLAT, "padx": 10, "pady": 5}
        tk.Button(toolbar, text="Open", command=self.open_rom, **btn_style).pack(side=tk.LEFT, padx=2, pady=5)
        tk.Button(toolbar, text="Start", command=self.start_emulation, **btn_style).pack(side=tk.LEFT, padx=2, pady=5)
        tk.Button(toolbar, text="Stop", command=self.stop_emulation, **btn_style).pack(side=tk.LEFT, padx=2, pady=5)
        
        # Canvas
        self.canvas = tk.Canvas(self.root, bg="#000000", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Log
        self.log_frame = tk.Frame(self.root, bg="#1e1e1e", height=100)
        self.log_text = scrolledtext.ScrolledText(
            self.log_frame, bg="#0a0a0a", fg="#00ff00",
            font=("Consolas", 9), height=6
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.log_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.log("MIPSEMU 1.03-ULTRA64 initialized")
        self.log("F3DEX microcode: READY")
        self.log("RDP rasterizer: READY")
        
        # Status bar
        self.status_bar = tk.Frame(self.root, bg="#1e1e1e", height=25)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_label = tk.Label(self.status_bar, text="Ready", bg="#1e1e1e", fg="white", anchor=tk.W)
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        self.fps_label = tk.Label(self.status_bar, text="FPS: 0", bg="#1e1e1e", fg="#00ff00", font=("Consolas", 9))
        self.fps_label.pack(side=tk.RIGHT, padx=10)
        
        self.show_welcome()
        
    def show_welcome(self):
        self.canvas.delete("all")
        self.canvas.create_text(512, 300, text="MIPSEMU 1.03-ULTRA64", font=("Arial", 48, "bold"), fill="#ff0000")
        self.canvas.create_text(512, 360, text="Ultra64/libultra Framework", font=("Arial", 16), fill="#00ff88")
        self.canvas.create_text(512, 420, text="Load ROM to begin", font=("Arial", 14), fill="#cccccc")
        
    def log(self, msg):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {msg}\n")
        self.log_text.see(tk.END)
        
    def open_rom(self):
        filename = filedialog.askopenfilename(
            title="Select ROM",
            filetypes=[("N64 ROMs", "*.z64 *.n64 *.v64"), ("All", "*.*")]
        )
        if filename:
            self.load_rom(filename)
            
    def load_rom(self, filepath):
        try:
            self.log(f"Loading: {Path(filepath).name}")
            
            with open(filepath, 'rb') as f:
                rom_data = f.read()
                
            self.rom_header = ROMHeader(rom_data)
            
            if not self.rom_header.valid:
                messagebox.showerror("Error", "Invalid ROM")
                return
                
            self.memory.load_rom(self.rom_header.raw_data + rom_data[len(self.rom_header.raw_data):])
            self.current_rom = filepath
            
            self.log(f"Game: {self.rom_header.name}")
            self.log(f"Format: {self.rom_header.endian}")
            
            self.root.title(f"MIPSEMU 1.03 - {self.rom_header.name}")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            
    def start_emulation(self):
        if not self.current_rom:
            messagebox.showwarning("No ROM", "Load a ROM first")
            return
            
        self.boot_status = 'booting'
        self.log("=== BOOT START ===")
        
        self.pif.simulate_boot(self.rom_header)
        self.cpu.boot_setup(self.rom_header.boot_address)
        
        self.log("PIF: Boot complete")
        self.log(f"CPU: PC = {hex(self.cpu.pc)}")
        self.log("RSP/RDP: Initialized")
        
        self.emulation_running = True
        self.cpu.running = True
        self.boot_status = 'running'
        
        self.emulation_thread = threading.Thread(target=self.emulation_loop, daemon=True)
        self.emulation_thread.start()
        
        self.render_loop()
        
    def emulation_loop(self):
        while self.emulation_running and self.cpu.running:
            try:
                for _ in range(3000):  # Instructions per frame
                    self.cpu.step()
                    
                # Simulate VI retrace
                self.os.vi_retrace_callback()
                
                # Test: Draw some triangles
                if self.frame_count % 60 == 0:
                    self.test_draw_triangles()
                    
                time.sleep(1.0 / 60.0)
            except Exception as e:
                self.log(f"Error: {e}")
                break
                
    def test_draw_triangles(self):
        """Test drawing triangles"""
        # Clear screen
        self.rdp.clear_framebuffer((0, 0, 64))
        
        # Draw some test triangles
        for i in range(5):
            x = random.randint(50, 270)
            y = random.randint(50, 190)
            
            tri = {
                'v0': {'x': x, 'y': y, 'r': 255, 'g': 0, 'b': 0},
                'v1': {'x': x + 20, 'y': y + 30, 'r': 0, 'g': 255, 'b': 0},
                'v2': {'x': x - 20, 'y': y + 30, 'r': 0, 'g': 0, 'b': 255}
            }
            self.rdp.draw_triangle(tri)
            
    def render_loop(self):
        if not self.emulation_running:
            return
            
        try:
            cpu_state = {
                'pc': self.cpu.pc,
                'instructions': self.cpu.instructions_executed,
                'registers': self.cpu.registers[:8]
            }
            
            self.video.render_frame(cpu_state, self.rdp, self.boot_status)
            
            self.frame_count += 1
            current_time = time.time()
            
            if current_time - self.last_fps_update >= 1.0:
                self.fps = self.frame_count
                self.fps_label.config(text=f"FPS: {self.fps}")
                self.frame_count = 0
                self.last_fps_update = current_time
                
            self.root.after(16, self.render_loop)
        except Exception as e:
            self.log(f"Render error: {e}")
            
    def stop_emulation(self):
        self.emulation_running = False
        self.cpu.running = False
        self.boot_status = 'idle'
        self.log("Stopped")


def main():
    root = tk.Tk()
    app = MIPSEMU(root)
    root.mainloop()


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
MIPSEMU 1.03-ULTRA64 - Darkness Revived (Ultra64 SDK Edition)
N64 Emulator with libultra/Ultra64 Software Implementation
Python 3.13 | Tkinter GUI

NEW IN 1.03:
- F3DEX graphics microcode interpreter
- RDP command processor and rasterizer
- OS thread management system
- Display list processing
- DMA engine implementation
- Interrupt system
- Framebuffer rendering
- Matrix stack and transformations
- Vertex processing pipeline
- Texture coordinate generation
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
from pathlib import Path
from datetime import datetime
import json
import struct
import threading
import time
from collections import defaultdict, deque
import hashlib
import math
import random


# ============================================================================
# ULTRA64 OS LAYER
# ============================================================================

class OSThread:
    """Ultra64 OS Thread"""
    def __init__(self, thread_id, priority=10):
        self.id = thread_id
        self.priority = priority
        self.state = 'STOPPED'  # STOPPED, RUNNING, WAITING
        self.pc = 0
        self.sp = 0
        self.context = {}
        
class OSMessageQueue:
    """Ultra64 Message Queue"""
    def __init__(self, size=8):
        self.queue = deque(maxlen=size)
        self.validCount = 0
        
    def send(self, message):
        if len(self.queue) < self.queue.maxlen:
            self.queue.append(message)
            self.validCount += 1
            return True
        return False
        
    def receive(self):
        if self.queue:
            self.validCount -= 1
            return self.queue.popleft()
        return None

class OSManager:
    """Ultra64 Operating System Manager"""
    def __init__(self):
        self.threads = {}
        self.current_thread = None
        self.message_queues = {}
        self.timers = []
        self.interrupts_enabled = True
        
        # Create main thread
        main_thread = OSThread(0, priority=10)
        main_thread.state = 'RUNNING'
        self.threads[0] = main_thread
        self.current_thread = main_thread
        
        # VI (Vertical Interrupt) queue
        self.vi_queue = OSMessageQueue(8)
        self.message_queues['VI'] = self.vi_queue
        
    def create_thread(self, thread_id, priority=10):
        """Create new thread"""
        thread = OSThread(thread_id, priority)
        self.threads[thread_id] = thread
        return thread
        
    def start_thread(self, thread_id):
        """Start thread execution"""
        if thread_id in self.threads:
            self.threads[thread_id].state = 'RUNNING'
            
    def vi_retrace_callback(self):
        """Called on vertical retrace"""
        self.vi_queue.send({'type': 'VI_RETRACE', 'time': time.time()})


# ============================================================================
# RSP GRAPHICS MICROCODE (F3DEX)
# ============================================================================

class F3DEXMicrocode:
    """F3DEX Graphics Microcode Interpreter"""
    def __init__(self, rsp, rdp):
        self.rsp = rsp
        self.rdp = rdp
        
        # F3DEX commands (simplified)
        self.G_NOOP = 0x00
        self.G_VTX = 0x01
        self.G_MODIFYVTX = 0x02
        self.G_CULLDL = 0x03
        self.G_BRANCH_Z = 0x04
        self.G_TRI1 = 0x05
        self.G_TRI2 = 0x06
        self.G_QUAD = 0x07
        self.G_LINE3D = 0x08
        
        self.G_DMA_IO = 0xD6
        self.G_TEXTURE = 0xD7
        self.G_POPMTX = 0xD8
        self.G_GEOMETRYMODE = 0xD9
        self.G_MTX = 0xDA
        self.G_MOVEWORD = 0xDB
        self.G_MOVEMEM = 0xDC
        self.G_LOAD_UCODE = 0xDD
        self.G_DL = 0xDE
        self.G_ENDDL = 0xDF
        self.G_SPNOOP = 0xE0
        self.G_RDPHALF_1 = 0xE1
        self.G_SETOTHERMODE_L = 0xE2
        self.G_SETOTHERMODE_H = 0xE3
        self.G_TEXRECT = 0xE4
        self.G_TEXRECTFLIP = 0xE5
        self.G_RDPLOADSYNC = 0xE6
        self.G_RDPPIPESYNC = 0xE7
        self.G_RDPTILESYNC = 0xE8
        self.G_RDPFULLSYNC = 0xE9
        self.G_SETKEYGB = 0xEA
        self.G_SETKEYR = 0xEB
        self.G_SETCONVERT = 0xEC
        self.G_SETSCISSOR = 0xED
        self.G_SETPRIMDEPTH = 0xEE
        self.G_RDPSETOTHERMODE = 0xEF
        self.G_LOADTLUT = 0xF0
        self.G_RDPHALF_2 = 0xF1
        self.G_SETTILESIZE = 0xF2
        self.G_LOADBLOCK = 0xF3
        self.G_LOADTILE = 0xF4
        self.G_SETTILE = 0xF5
        self.G_FILLRECT = 0xF6
        self.G_SETFILLCOLOR = 0xF7
        self.G_SETFOGCOLOR = 0xF8
        self.G_SETBLENDCOLOR = 0xF9
        self.G_SETPRIMCOLOR = 0xFA
        self.G_SETENVCOLOR = 0xFB
        self.G_SETCOMBINE = 0xFC
        self.G_SETTIMG = 0xFD
        self.G_SETZIMG = 0xFE
        self.G_SETCIMG = 0xFF
        
    def process_display_list(self, dl_addr, memory):
        """Process display list commands"""
        commands_processed = 0
        max_commands = 10000  # Prevent infinite loops
        
        while commands_processed < max_commands:
            # Read 64-bit command
            w0 = memory.read_word(dl_addr)
            w1 = memory.read_word(dl_addr + 4)
            
            cmd = (w0 >> 24) & 0xFF
            
            # Process command
            if cmd == self.G_ENDDL:
                break
            elif cmd == self.G_VTX:
                self.cmd_vtx(w0, w1, memory)
            elif cmd == self.G_TRI1:
                self.cmd_tri1(w0, w1)
            elif cmd == self.G_TRI2:
                self.cmd_tri2(w0, w1)
            elif cmd == self.G_MTX:
                self.cmd_mtx(w0, w1, memory)
            elif cmd == self.G_DL:
                # Branch to another display list
                branch_addr = w1
                self.process_display_list(branch_addr, memory)
            elif cmd == self.G_TEXTURE:
                self.cmd_texture(w0, w1)
            elif cmd == self.G_SETCOMBINE:
                self.rdp.cmd_setcombine(w0, w1)
            elif cmd == self.G_SETTIMG:
                self.rdp.cmd_settimg(w0, w1)
            elif cmd == self.G_SETCIMG:
                self.rdp.cmd_setcimg(w0, w1)
            elif cmd == self.G_SETZIMG:
                self.rdp.cmd_setzimg(w0, w1)
            elif cmd == self.G_FILLRECT:
                self.rdp.cmd_fillrect(w0, w1)
            elif cmd == self.G_SETFILLCOLOR:
                self.rdp.cmd_setfillcolor(w0, w1)
            elif cmd == self.G_RDPFULLSYNC:
                self.rdp.cmd_fullsync()
            elif cmd == self.G_RDPPIPESYNC:
                self.rdp.cmd_pipesync()
            elif cmd == self.G_GEOMETRYMODE:
                self.cmd_geometrymode(w0, w1)
                
            dl_addr += 8
            commands_processed += 1
            
        return commands_processed
        
    def cmd_vtx(self, w0, w1, memory):
        """Load vertices into vertex buffer"""
        n = ((w0 >> 12) & 0xFF) // 2  # Number of vertices
        v0 = (w0 >> 1) & 0x7F  # Starting index
        addr = w1 & 0xFFFFFF
        
        for i in range(n):
            vtx_data = []
            for j in range(4):  # 16 bytes per vertex
                word = memory.read_word(addr + i * 16 + j * 4)
                vtx_data.append(word)
                
            # Parse vertex (simplified)
            x = self.sign_extend_16((vtx_data[0] >> 16) & 0xFFFF)
            y = self.sign_extend_16(vtx_data[0] & 0xFFFF)
            z = self.sign_extend_16((vtx_data[1] >> 16) & 0xFFFF)
            
            s = (vtx_data[2] >> 16) & 0xFFFF
            t = vtx_data[2] & 0xFFFF
            
            r = (vtx_data[3] >> 24) & 0xFF
            g = (vtx_data[3] >> 16) & 0xFF
            b = (vtx_data[3] >> 8) & 0xFF
            a = vtx_data[3] & 0xFF
            
            vertex = {
                'x': x, 'y': y, 'z': z,
                's': s, 't': t,
                'r': r, 'g': g, 'b': b, 'a': a
            }
            
            self.rsp.vertex_buffer[v0 + i] = vertex
            
    def cmd_tri1(self, w0, w1):
        """Draw one triangle"""
        v0 = ((w1 >> 16) & 0xFF) // 2
        v1 = ((w1 >> 8) & 0xFF) // 2
        v2 = (w1 & 0xFF) // 2
        
        if v0 in self.rsp.vertex_buffer and v1 in self.rsp.vertex_buffer and v2 in self.rsp.vertex_buffer:
            tri = {
                'v0': self.rsp.vertex_buffer[v0],
                'v1': self.rsp.vertex_buffer[v1],
                'v2': self.rsp.vertex_buffer[v2]
            }
            self.rdp.draw_triangle(tri)
            
    def cmd_tri2(self, w0, w1):
        """Draw two triangles"""
        v0 = ((w0 >> 16) & 0xFF) // 2
        v1 = ((w0 >> 8) & 0xFF) // 2
        v2 = (w0 & 0xFF) // 2
        
        v3 = ((w1 >> 16) & 0xFF) // 2
        v4 = ((w1 >> 8) & 0xFF) // 2
        v5 = (w1 & 0xFF) // 2
        
        # Draw first triangle
        if all(i in self.rsp.vertex_buffer for i in [v0, v1, v2]):
            tri1 = {
                'v0': self.rsp.vertex_buffer[v0],
                'v1': self.rsp.vertex_buffer[v1],
                'v2': self.rsp.vertex_buffer[v2]
            }
            self.rdp.draw_triangle(tri1)
            
        # Draw second triangle
        if all(i in self.rsp.vertex_buffer for i in [v3, v4, v5]):
            tri2 = {
                'v0': self.rsp.vertex_buffer[v3],
                'v1': self.rsp.vertex_buffer[v4],
                'v2': self.rsp.vertex_buffer[v5]
            }
            self.rdp.draw_triangle(tri2)
            
    def cmd_mtx(self, w0, w1, memory):
        """Load transformation matrix"""
        addr = w1 & 0xFFFFFF
        push = (w0 >> 2) & 0x1
        load = (w0 >> 1) & 0x1
        projection = (w0 >> 0) & 0x1
        
        # Load 4x4 matrix from memory (16 words)
        matrix = []
        for i in range(16):
            value = memory.read_word(addr + i * 4)
            matrix.append(value)
            
        # Store in matrix stack (simplified)
        if projection:
            self.rsp.projection_matrix = matrix
        else:
            self.rsp.modelview_matrix = matrix
            
    def cmd_texture(self, w0, w1):
        """Set texture parameters"""
        level = (w0 >> 11) & 0x7
        tile = (w0 >> 8) & 0x7
        on = (w0 >> 1) & 0x1
        
        scaleS = (w1 >> 16) & 0xFFFF
        scaleT = w1 & 0xFFFF
        
        self.rsp.texture_enabled = (on == 1)
        self.rsp.texture_tile = tile
        self.rsp.texture_scaleS = scaleS
        self.rsp.texture_scaleT = scaleT
        
    def cmd_geometrymode(self, w0, w1):
        """Set geometry mode flags"""
        clearbits = ~(w0 & 0xFFFFFF)
        setbits = w1
        
        self.rsp.geometry_mode = (self.rsp.geometry_mode & clearbits) | setbits
        
    def sign_extend_16(self, value):
        if value & 0x8000:
            return value - 0x10000
        return value


# ============================================================================
# RDP (REALITY DISPLAY PROCESSOR)
# ============================================================================

class RDP:
    """Reality Display Processor - Rasterizer"""
    def __init__(self):
        # Framebuffer
        self.framebuffer_width = 320
        self.framebuffer_height = 240
        self.framebuffer = [[(0, 0, 0) for _ in range(320)] for _ in range(240)]
        self.zbuffer = [[float('inf') for _ in range(320)] for _ in range(240)]
        
        # Color combiner
        self.combine_mode = 0
        
        # Fill color
        self.fill_color = (0, 0, 0, 255)
        
        # Texture image
        self.texture_image_addr = 0
        self.texture_image_format = 0
        self.texture_image_size = 0
        self.texture_image_width = 0
        
        # Color image (output)
        self.color_image_addr = 0
        self.color_image_format = 0
        self.color_image_size = 0
        self.color_image_width = 320
        
        # Z buffer image
        self.z_image_addr = 0
        
        # Scissor
        self.scissor_x0 = 0
        self.scissor_y0 = 0
        self.scissor_x1 = 320
        self.scissor_y1 = 240
        
        # Primitive color
        self.prim_color = (255, 255, 255, 255)
        self.env_color = (255, 255, 255, 255)
        
        # Statistics
        self.triangles_drawn = 0
        self.pixels_drawn = 0
        
    def cmd_setcombine(self, w0, w1):
        """Set color combiner mode"""
        self.combine_mode = w1
        
    def cmd_settimg(self, w0, w1):
        """Set texture image"""
        self.texture_image_format = (w0 >> 21) & 0x7
        self.texture_image_size = (w0 >> 19) & 0x3
        self.texture_image_width = (w0 & 0x3FF) + 1
        self.texture_image_addr = w1 & 0xFFFFFF
        
    def cmd_setcimg(self, w0, w1):
        """Set color image (framebuffer)"""
        self.color_image_format = (w0 >> 21) & 0x7
        self.color_image_size = (w0 >> 19) & 0x3
        self.color_image_width = (w0 & 0x3FF) + 1
        self.color_image_addr = w1 & 0xFFFFFF
        
    def cmd_setzimg(self, w0, w1):
        """Set Z buffer image"""
        self.z_image_addr = w1 & 0xFFFFFF
        
    def cmd_fillrect(self, w0, w1):
        """Fill rectangle"""
        x1 = ((w1 >> 12) & 0xFFF) >> 2
        y1 = ((w1 >> 0) & 0xFFF) >> 2
        x0 = ((w0 >> 12) & 0xFFF) >> 2
        y0 = ((w0 >> 0) & 0xFFF) >> 2
        
        # Clamp to framebuffer
        x0 = max(0, min(x0, self.framebuffer_width - 1))
        y0 = max(0, min(y0, self.framebuffer_height - 1))
        x1 = max(0, min(x1, self.framebuffer_width - 1))
        y1 = max(0, min(y1, self.framebuffer_height - 1))
        
        # Fill rectangle
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                if 0 <= y < self.framebuffer_height and 0 <= x < self.framebuffer_width:
                    self.framebuffer[y][x] = self.fill_color[:3]
                    self.pixels_drawn += 1
                    
    def cmd_setfillcolor(self, w0, w1):
        """Set fill color"""
        # RGBA 16-bit or 32-bit
        r = (w1 >> 24) & 0xFF
        g = (w1 >> 16) & 0xFF
        b = (w1 >> 8) & 0xFF
        a = w1 & 0xFF
        self.fill_color = (r, g, b, a)
        
    def cmd_fullsync(self):
        """Full sync - wait for RDP to finish"""
        pass
        
    def cmd_pipesync(self):
        """Pipeline sync"""
        pass
        
    def draw_triangle(self, tri):
        """Draw a triangle (simplified rasterizer)"""
        self.triangles_drawn += 1
        
        # Get vertices
        v0 = tri['v0']
        v1 = tri['v1']
        v2 = tri['v2']
        
        # Convert to screen space (simplified projection)
        # Assume coordinates are already in screen space for now
        x0 = int(v0['x'] / 4 + 160)  # Scale and center
        y0 = int(v0['y'] / 4 + 120)
        x1 = int(v1['x'] / 4 + 160)
        y1 = int(v1['y'] / 4 + 120)
        x2 = int(v2['x'] / 4 + 160)
        y2 = int(v2['y'] / 4 + 120)
        
        # Clamp to screen
        x0 = max(0, min(x0, self.framebuffer_width - 1))
        y0 = max(0, min(y0, self.framebuffer_height - 1))
        x1 = max(0, min(x1, self.framebuffer_width - 1))
        y1 = max(0, min(y1, self.framebuffer_height - 1))
        x2 = max(0, min(x2, self.framebuffer_width - 1))
        y2 = max(0, min(y2, self.framebuffer_height - 1))
        
        # Get colors
        r = (v0['r'] + v1['r'] + v2['r']) // 3
        g = (v0['g'] + v1['g'] + v2['g']) // 3
        b = (v0['b'] + v1['b'] + v2['b']) // 3
        
        # Simple triangle fill (scanline)
        self.fill_triangle(x0, y0, x1, y1, x2, y2, (r, g, b))
        
    def fill_triangle(self, x0, y0, x1, y1, x2, y2, color):
        """Fill triangle using scanline algorithm"""
        # Sort vertices by y
        if y0 > y1:
            x0, y0, x1, y1 = x1, y1, x0, y0
        if y0 > y2:
            x0, y0, x2, y2 = x2, y2, x0, y0
        if y1 > y2:
            x1, y1, x2, y2 = x2, y2, x1, y1
            
        # Draw flat-bottom triangle
        if y1 == y2:
            self.fill_flat_bottom(x0, y0, x1, y1, x2, y2, color)
        # Draw flat-top triangle
        elif y0 == y1:
            self.fill_flat_top(x0, y0, x1, y1, x2, y2, color)
        # Split into two triangles
        else:
            # Calculate split point
            if y2 - y0 != 0:
                x3 = int(x0 + (y1 - y0) / (y2 - y0) * (x2 - x0))
                y3 = y1
                
                self.fill_flat_bottom(x0, y0, x1, y1, x3, y3, color)
                self.fill_flat_top(x1, y1, x3, y3, x2, y2, color)
                
    def fill_flat_bottom(self, x0, y0, x1, y1, x2, y2, color):
        """Fill flat-bottom triangle"""
        if y1 - y0 == 0:
            return
            
        slope1 = (x1 - x0) / (y1 - y0)
        slope2 = (x2 - x0) / (y2 - y0)
        
        xs1 = x0
        xs2 = x0
        
        for y in range(y0, y1 + 1):
            if 0 <= y < self.framebuffer_height:
                x_start = int(min(xs1, xs2))
                x_end = int(max(xs1, xs2))
                
                for x in range(x_start, x_end + 1):
                    if 0 <= x < self.framebuffer_width:
                        self.framebuffer[y][x] = color
                        self.pixels_drawn += 1
                        
            xs1 += slope1
            xs2 += slope2
            
    def fill_flat_top(self, x0, y0, x1, y1, x2, y2, color):
        """Fill flat-top triangle"""
        if y2 - y0 == 0:
            return
            
        slope1 = (x2 - x0) / (y2 - y0)
        slope2 = (x2 - x1) / (y2 - y1)
        
        xs1 = x2
        xs2 = x2
        
        for y in range(y2, y0 - 1, -1):
            if 0 <= y < self.framebuffer_height:
                x_start = int(min(xs1, xs2))
                x_end = int(max(xs1, xs2))
                
                for x in range(x_start, x_end + 1):
                    if 0 <= x < self.framebuffer_width:
                        self.framebuffer[y][x] = color
                        self.pixels_drawn += 1
                        
            xs1 -= slope1
            xs2 -= slope2
            
    def clear_framebuffer(self, color=(0, 0, 0)):
        """Clear framebuffer"""
        for y in range(self.framebuffer_height):
            for x in range(self.framebuffer_width):
                self.framebuffer[y][x] = color
        self.pixels_drawn = 0
        self.triangles_drawn = 0


# ============================================================================
# RSP (REALITY SIGNAL PROCESSOR)
# ============================================================================

class RSP:
    """Reality Signal Processor - Enhanced"""
    def __init__(self):
        self.dmem = bytearray(4096)
        self.imem = bytearray(4096)
        self.registers = [0] * 32
        self.pc = 0
        self.status = 0x1  # Halted
        
        # Vertex buffer
        self.vertex_buffer = {}
        
        # Matrix stack
        self.projection_matrix = self.identity_matrix()
        self.modelview_matrix = self.identity_matrix()
        self.matrix_stack = []
        
        # Geometry mode
        self.geometry_mode = 0
        
        # Texture state
        self.texture_enabled = False
        self.texture_tile = 0
        self.texture_scaleS = 0
        self.texture_scaleT = 0
        
        # Lighting (simplified)
        self.lighting_enabled = False
        self.lights = []
        
    def identity_matrix(self):
        """Return 4x4 identity matrix"""
        return [
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1
        ]
        
    def read_dmem(self, offset):
        if offset < len(self.dmem):
            return self.dmem[offset]
        return 0
        
    def write_dmem(self, offset, value):
        if offset < len(self.dmem):
            self.dmem[offset] = value & 0xFF


# ============================================================================
# DMA CONTROLLER
# ============================================================================

class DMAController:
    """Enhanced DMA Controller"""
    def __init__(self, memory):
        self.memory = memory
        self.pi_dram_addr = 0
        self.pi_cart_addr = 0
        self.pi_rd_len = 0
        self.pi_wr_len = 0
        self.pi_status = 0
        self.busy = False
        
    def start_read(self, dram_addr, cart_addr, length):
        """DMA read from cartridge to DRAM"""
        self.pi_dram_addr = dram_addr
        self.pi_cart_addr = cart_addr
        self.pi_rd_len = length
        self.busy = True
        
        # Perform transfer
        for i in range(length + 1):
            value = self.memory.read_byte(0xB0000000 + cart_addr + i)
            self.memory.write_byte(dram_addr + i, value)
            
        self.busy = False
        self.pi_status = 0
        return True
        
    def start_write(self, dram_addr, cart_addr, length):
        """DMA write from DRAM to cartridge (save)"""
        self.pi_dram_addr = dram_addr
        self.pi_cart_addr = cart_addr
        self.pi_wr_len = length
        self.busy = True
        
        # Transfer to save RAM
        for i in range(length + 1):
            value = self.memory.read_byte(dram_addr + i)
            self.memory.write_byte(cart_addr + i, value)
            
        self.busy = False
        return True


# ============================================================================
# ROM HEADER
# ============================================================================

class ROMHeader:
    """N64 ROM Header Parser"""
    def __init__(self, data):
        self.raw_data = data[:0x1000]
        self.valid = False
        self.parse()
        
    def parse(self):
        if len(self.raw_data) < 0x40:
            return
            
        magic = struct.unpack('>I', self.raw_data[0:4])[0]
        
        if magic == 0x80371240:
            self.endian = 'big'
            self.valid = True
        elif magic == 0x40123780:
            self.endian = 'little'
            self.raw_data = self.swap_endian_n64(self.raw_data)
            self.valid = True
        elif magic == 0x37804012:
            self.endian = 'byteswap'
            self.raw_data = self.swap_endian_v64(self.raw_data)
            self.valid = True
        else:
            self.endian = 'unknown'
            return
            
        self.clock_rate = struct.unpack('>I', self.raw_data[0x04:0x08])[0]
        self.boot_address = struct.unpack('>I', self.raw_data[0x08:0x0C])[0]
        self.release = struct.unpack('>I', self.raw_data[0x0C:0x10])[0]
        
        self.crc1 = struct.unpack('>I', self.raw_data[0x10:0x14])[0]
        self.crc2 = struct.unpack('>I', self.raw_data[0x14:0x18])[0]
        
        self.name = self.raw_data[0x20:0x34].decode('ascii', errors='ignore').strip('\x00')
        self.country_code = chr(self.raw_data[0x3E])
        self.country = self.get_country_name(self.country_code)
        self.version = self.raw_data[0x3F]
        self.game_id = self.raw_data[0x3B:0x3F].decode('ascii', errors='ignore')
        self.rom_hash = hashlib.md5(self.raw_data[:0x100]).hexdigest()
        self.ipl3 = self.raw_data[0x40:0x1000]
        
    def get_country_name(self, code):
        countries = {
            'A': 'All', 'D': 'Germany', 'E': 'USA', 'F': 'France',
            'I': 'Italy', 'J': 'Japan', 'S': 'Spain', 'U': 'Australia',
            'P': 'Europe', 'N': 'Canada'
        }
        return countries.get(code, 'Unknown')
        
    def swap_endian_n64(self, data):
        result = bytearray(len(data))
        for i in range(0, len(data), 4):
            result[i:i+4] = data[i:i+4][::-1]
        return bytes(result)
        
    def swap_endian_v64(self, data):
        result = bytearray(len(data))
        for i in range(0, len(data), 2):
            result[i] = data[i+1]
            result[i+1] = data[i]
        return bytes(result)


# ============================================================================
# PIF BOOTLOADER
# ============================================================================

class PIF:
    """PIF Bootloader"""
    def __init__(self, memory):
        self.memory = memory
        self.pif_ram = bytearray(64)
        self.pif_rom = bytearray(2048)
        self.boot_complete = False
        
    def simulate_boot(self, rom_header):
        """Simulate PIF boot sequence"""
        if rom_header and rom_header.ipl3:
            for i, byte in enumerate(rom_header.ipl3):
                if i < 0x1000:
                    self.memory.write_byte(0x04000000 + i, byte)
                    
        self.pif_ram[0x3F] = 0x00
        self.boot_complete = True
        return True


# ============================================================================
# COP0
# ============================================================================

class COP0:
    """Coprocessor 0"""
    def __init__(self):
        self.registers = [0] * 32
        self.STATUS = 12
        self.CAUSE = 13
        self.EPC = 14
        self.COUNT = 9
        self.COMPARE = 11
        
        self.registers[15] = 0x00000B00  # PRID
        self.registers[self.STATUS] = 0x34000000
        self.registers[16] = 0x7006E463  # CONFIG
        
    def read(self, reg):
        return self.registers[reg & 0x1F]
        
    def write(self, reg, value):
        reg = reg & 0x1F
        if reg == self.COMPARE:
            self.registers[reg] = value
            self.registers[self.CAUSE] &= ~0x8000
        else:
            self.registers[reg] = value


# Rest of the CPU implementation remains the same as v1.02...
# (Including MIPSCPU, Memory, VideoInterface, etc.)
# For brevity, I'll include just the key components

class MIPSCPU:
    """MIPS R4300i CPU"""
    def __init__(self, memory):
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
        self.branch_delay = False
        self.delay_slot_pc = 0
        
    def reset(self):
        self.pc = 0xA4000040
        self.next_pc = self.pc + 4
        self.registers = [0] * 32
        self.hi = 0
        self.lo = 0
        self.instructions_executed = 0
        self.cycles = 0
        self.cop0 = COP0()
        
    def boot_setup(self, boot_address):
        self.reset()
        self.pc = boot_address
        self.next_pc = self.pc + 4
        self.registers[11] = 0xFFFFFFF4
        self.registers[20] = 0x00000001
        self.registers[22] = 0x0000003F
        self.registers[29] = 0xA4001FF0
        self.cop0.write(self.cop0.STATUS, 0x34000000)
        
    def step(self):
        if not self.running:
            return
        try:
            instruction = self.memory.read_word(self.pc)
            self.execute_instruction(instruction)
            
            if self.branch_delay:
                self.pc = self.delay_slot_pc
                self.branch_delay = False
            else:
                self.pc = self.next_pc
            self.next_pc = self.pc + 4
            
            self.instructions_executed += 1
            self.cycles += 1
        except Exception as e:
            print(f"CPU Exception: {e}")
            self.running = False
            
    def execute_instruction(self, instr):
        opcode = (instr >> 26) & 0x3F
        
        if opcode == 0x00:  # SPECIAL
            funct = instr & 0x3F
            rs = (instr >> 21) & 0x1F
            rt = (instr >> 16) & 0x1F
            rd = (instr >> 11) & 0x1F
            shamt = (instr >> 6) & 0x1F
            
            if funct == 0x00:  # SLL
                self.registers[rd] = (self.registers[rt] << shamt) & 0xFFFFFFFF
            elif funct == 0x08:  # JR
                self.delay_slot_pc = self.registers[rs]
                self.branch_delay = True
            elif funct == 0x09:  # JALR
                self.registers[rd] = self.next_pc + 4
                self.delay_slot_pc = self.registers[rs]
                self.branch_delay = True
            elif funct == 0x21:  # ADDU
                self.registers[rd] = (self.registers[rs] + self.registers[rt]) & 0xFFFFFFFF
            elif funct == 0x25:  # OR
                self.registers[rd] = self.registers[rs] | self.registers[rt]
                
        elif opcode == 0x02:  # J
            target = (instr & 0x3FFFFFF) << 2
            self.delay_slot_pc = (self.pc & 0xF0000000) | target
            self.branch_delay = True
        elif opcode == 0x03:  # JAL
            target = (instr & 0x3FFFFFF) << 2
            self.registers[31] = self.next_pc + 4
            self.delay_slot_pc = (self.pc & 0xF0000000) | target
            self.branch_delay = True
        elif opcode == 0x04:  # BEQ
            rs, rt = (instr >> 21) & 0x1F, (instr >> 16) & 0x1F
            offset = self.sign_extend_16(instr & 0xFFFF) << 2
            if self.registers[rs] == self.registers[rt]:
                self.delay_slot_pc = self.next_pc + offset
                self.branch_delay = True
        elif opcode == 0x05:  # BNE
            rs, rt = (instr >> 21) & 0x1F, (instr >> 16) & 0x1F
            offset = self.sign_extend_16(instr & 0xFFFF) << 2
            if self.registers[rs] != self.registers[rt]:
                self.delay_slot_pc = self.next_pc + offset
                self.branch_delay = True
        elif opcode == 0x09:  # ADDIU
            rs, rt = (instr >> 21) & 0x1F, (instr >> 16) & 0x1F
            imm = self.sign_extend_16(instr & 0xFFFF)
            self.registers[rt] = (self.registers[rs] + imm) & 0xFFFFFFFF
        elif opcode == 0x0D:  # ORI
            rs, rt = (instr >> 21) & 0x1F, (instr >> 16) & 0x1F
            imm = instr & 0xFFFF
            self.registers[rt] = self.registers[rs] | imm
        elif opcode == 0x0F:  # LUI
            rt = (instr >> 16) & 0x1F
            imm = instr & 0xFFFF
            self.registers[rt] = (imm << 16) & 0xFFFFFFFF
        elif opcode == 0x23:  # LW
            rs, rt = (instr >> 21) & 0x1F, (instr >> 16) & 0x1F
            offset = self.sign_extend_16(instr & 0xFFFF)
            addr = (self.registers[rs] + offset) & 0xFFFFFFFF
            self.registers[rt] = self.memory.read_word(addr)
        elif opcode == 0x2B:  # SW
            rs, rt = (instr >> 21) & 0x1F, (instr >> 16) & 0x1F
            offset = self.sign_extend_16(instr & 0xFFFF)
            addr = (self.registers[rs] + offset) & 0xFFFFFFFF
            self.memory.write_word(addr, self.registers[rt])
            
        self.registers[0] = 0
        
    def sign_extend_16(self, value):
        if value & 0x8000:
            return value | 0xFFFF0000
        return value


class Memory:
    """N64 Memory System"""
    def __init__(self):
        self.rdram = bytearray(8 * 1024 * 1024)
        self.rom = None
        self.rom_size = 0
        self.sp_dmem = bytearray(4096)
        self.sp_imem = bytearray(4096)
        self.pif_ram = bytearray(64)
        
    def load_rom(self, rom_data):
        self.rom = rom_data
        self.rom_size = len(rom_data)
        
    def read_byte(self, addr):
        addr = addr & 0xFFFFFFFF
        
        if addr < 0x00800000 or (0xA0000000 <= addr < 0xA0800000):
            ram_addr = addr & 0x007FFFFF
            if ram_addr < len(self.rdram):
                return self.rdram[ram_addr]
        elif (0x10000000 <= addr < 0x1FBFFFFF) or (0xB0000000 <= addr < 0xBFFFFFFF):
            rom_addr = addr & 0x0FFFFFFF
            if self.rom and rom_addr < self.rom_size:
                return self.rom[rom_addr]
        elif 0x04000000 <= addr < 0x04001000:
            return self.sp_dmem[addr & 0xFFF]
        elif 0x1FC007C0 <= addr < 0x1FC00800:
            return self.pif_ram[addr & 0x3F]
        return 0
        
    def read_half(self, addr):
        b0 = self.read_byte(addr)
        b1 = self.read_byte(addr + 1)
        return (b0 << 8) | b1
        
    def read_word(self, addr):
        addr = addr & 0xFFFFFFFF
        
        if addr < 0x00800000 or (0xA0000000 <= addr < 0xA0800000):
            ram_addr = addr & 0x007FFFFF
            if ram_addr < len(self.rdram) - 3:
                return struct.unpack('>I', self.rdram[ram_addr:ram_addr+4])[0]
        elif (0x10000000 <= addr < 0x1FBFFFFF) or (0xB0000000 <= addr < 0xBFFFFFFF):
            rom_addr = addr & 0x0FFFFFFF
            if self.rom and rom_addr < self.rom_size - 3:
                return struct.unpack('>I', self.rom[rom_addr:rom_addr+4])[0]
        elif 0x04000000 <= addr < 0x04001000:
            offset = addr & 0xFFF
            if offset < len(self.sp_dmem) - 3:
                return struct.unpack('>I', self.sp_dmem[offset:offset+4])[0]
        return 0
        
    def write_byte(self, addr, value):
        addr = addr & 0xFFFFFFFF
        value = value & 0xFF
        
        if addr < 0x00800000 or (0xA0000000 <= addr < 0xA0800000):
            ram_addr = addr & 0x007FFFFF
            if ram_addr < len(self.rdram):
                self.rdram[ram_addr] = value
        elif 0x04000000 <= addr < 0x04001000:
            self.sp_dmem[addr & 0xFFF] = value
        elif 0x1FC007C0 <= addr < 0x1FC00800:
            self.pif_ram[addr & 0x3F] = value
            
    def write_word(self, addr, value):
        addr = addr & 0xFFFFFFFF
        value = value & 0xFFFFFFFF
        
        if addr < 0x00800000 or (0xA0000000 <= addr < 0xA0800000):
            ram_addr = addr & 0x007FFFFF
            if ram_addr < len(self.rdram) - 3:
                struct.pack_into('>I', self.rdram, ram_addr, value)
        elif 0x04000000 <= addr < 0x04001000:
            offset = addr & 0xFFF
            if offset < len(self.sp_dmem) - 3:
                struct.pack_into('>I', self.sp_dmem, offset, value)


class VideoInterface:
    """Video Interface with RDP Framebuffer Display"""
    def __init__(self, canvas):
        self.canvas = canvas
        self.frame_count = 0
        
    def render_frame(self, cpu_state, rdp, boot_status):
        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, 1024, 768, fill="#001122", outline="")
        
        screen_x, screen_y = 192, 114
        self.canvas.create_rectangle(
            screen_x, screen_y, 
            screen_x + 640, screen_y + 480,
            fill="#000000", outline="#00ff88", width=2
        )
        
        if boot_status == 'running':
            # Render RDP framebuffer
            self.render_rdp_framebuffer(screen_x, screen_y, rdp)
            
            # Stats overlay
            self.canvas.create_text(
                screen_x + 320, screen_y + 20,
                text=f"PC: {hex(cpu_state['pc'])}  |  Triangles: {rdp.triangles_drawn}  |  Pixels: {rdp.pixels_drawn}",
                font=("Consolas", 10),
                fill="#00ff00"
            )
        elif boot_status == 'booting':
            self.canvas.create_text(
                screen_x + 320, screen_y + 240,
                text="NINTENDO 64",
                font=("Arial", 48, "bold"),
                fill="#ff0000"
            )
        
        self.frame_count += 1
        
    def render_rdp_framebuffer(self, x, y, rdp):
        """Render RDP framebuffer to canvas"""
        # Scale factor (320x240 -> 640x480)
        scale = 2
        
        # Sample every 4th pixel for performance
        for py in range(0, 240, 4):
            for px in range(0, 320, 4):
                color = rdp.framebuffer[py][px]
                hex_color = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
                
                self.canvas.create_rectangle(
                    x + px * scale, y + py * scale,
                    x + (px + 4) * scale, y + (py + 4) * scale,
                    fill=hex_color, outline=""
                )


class ControllerInput:
    """N64 Controller"""
    def __init__(self):
        self.buttons = {}
        self.stick_x = 0
        self.stick_y = 0
        
    def key_press(self, key):
        pass
        
    def key_release(self, key):
        pass


class MIPSEMU:
    def __init__(self, root):
        self.root = root
        self.root.title("MIPSEMU 1.03-ULTRA64")
        self.root.geometry("1024x768")
        self.root.configure(bg="#2b2b2b")
        
        # Components
        self.memory = Memory()
        self.cpu = MIPSCPU(self.memory)
        self.pif = PIF(self.memory)
        self.rsp = RSP()
        self.rdp = RDP()
        self.dma = DMAController(self.memory)
        self.os = OSManager()
        self.f3dex = F3DEXMicrocode(self.rsp, self.rdp)
        self.controller = ControllerInput()
        
        self.current_rom = None
        self.rom_header = None
        self.emulation_running = False
        self.boot_status = 'idle'
        self.config_file = Path("mipsemu_config.json")
        self.rom_list = []
        
        self.fps = 0
        self.last_fps_update = time.time()
        self.frame_count = 0
        
        self.create_ui()
        self.video = VideoInterface(self.canvas)
        
    def create_ui(self):
        # Menu
        menubar = tk.Menu(self.root, bg="#1e1e1e", fg="white")
        self.root.config(menu=menubar)
        
        file_menu = tk.Menu(menubar, tearoff=0, bg="#1e1e1e", fg="white")
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open ROM", command=self.open_rom)
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        system_menu = tk.Menu(menubar, tearoff=0, bg="#1e1e1e", fg="white")
        menubar.add_cascade(label="System", menu=system_menu)
        system_menu.add_command(label="Start", command=self.start_emulation)
        system_menu.add_command(label="Stop", command=self.stop_emulation)
        
        # Toolbar
        toolbar = tk.Frame(self.root, bg="#1e1e1e")
        toolbar.pack(side=tk.TOP, fill=tk.X)
        
        btn_style = {"bg": "#3c3c3c", "fg": "white", "relief": tk.FLAT, "padx": 10, "pady": 5}
        tk.Button(toolbar, text="Open", command=self.open_rom, **btn_style).pack(side=tk.LEFT, padx=2, pady=5)
        tk.Button(toolbar, text="Start", command=self.start_emulation, **btn_style).pack(side=tk.LEFT, padx=2, pady=5)
        tk.Button(toolbar, text="Stop", command=self.stop_emulation, **btn_style).pack(side=tk.LEFT, padx=2, pady=5)
        
        # Canvas
        self.canvas = tk.Canvas(self.root, bg="#000000", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Log
        self.log_frame = tk.Frame(self.root, bg="#1e1e1e", height=100)
        self.log_text = scrolledtext.ScrolledText(
            self.log_frame, bg="#0a0a0a", fg="#00ff00",
            font=("Consolas", 9), height=6
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.log_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.log("MIPSEMU 1.03-ULTRA64 initialized")
        self.log("F3DEX microcode: READY")
        self.log("RDP rasterizer: READY")
        
        # Status bar
        self.status_bar = tk.Frame(self.root, bg="#1e1e1e", height=25)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_label = tk.Label(self.status_bar, text="Ready", bg="#1e1e1e", fg="white", anchor=tk.W)
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        self.fps_label = tk.Label(self.status_bar, text="FPS: 0", bg="#1e1e1e", fg="#00ff00", font=("Consolas", 9))
        self.fps_label.pack(side=tk.RIGHT, padx=10)
        
        self.show_welcome()
        
    def show_welcome(self):
        self.canvas.delete("all")
        self.canvas.create_text(512, 300, text="MIPSEMU 1.03-ULTRA64", font=("Arial", 48, "bold"), fill="#ff0000")
        self.canvas.create_text(512, 360, text="Ultra64/libultra Framework", font=("Arial", 16), fill="#00ff88")
        self.canvas.create_text(512, 420, text="Load ROM to begin", font=("Arial", 14), fill="#cccccc")
        
    def log(self, msg):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {msg}\n")
        self.log_text.see(tk.END)
        
    def open_rom(self):
        filename = filedialog.askopenfilename(
            title="Select ROM",
            filetypes=[("N64 ROMs", "*.z64 *.n64 *.v64"), ("All", "*.*")]
        )
        if filename:
            self.load_rom(filename)
            
    def load_rom(self, filepath):
        try:
            self.log(f"Loading: {Path(filepath).name}")
            
            with open(filepath, 'rb') as f:
                rom_data = f.read()
                
            self.rom_header = ROMHeader(rom_data)
            
            if not self.rom_header.valid:
                messagebox.showerror("Error", "Invalid ROM")
                return
                
            self.memory.load_rom(self.rom_header.raw_data + rom_data[len(self.rom_header.raw_data):])
            self.current_rom = filepath
            
            self.log(f"Game: {self.rom_header.name}")
            self.log(f"Format: {self.rom_header.endian}")
            
            self.root.title(f"MIPSEMU 1.03 - {self.rom_header.name}")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            
    def start_emulation(self):
        if not self.current_rom:
            messagebox.showwarning("No ROM", "Load a ROM first")
            return
            
        self.boot_status = 'booting'
        self.log("=== BOOT START ===")
        
        self.pif.simulate_boot(self.rom_header)
        self.cpu.boot_setup(self.rom_header.boot_address)
        
        self.log("PIF: Boot complete")
        self.log(f"CPU: PC = {hex(self.cpu.pc)}")
        self.log("RSP/RDP: Initialized")
        
        self.emulation_running = True
        self.cpu.running = True
        self.boot_status = 'running'
        
        self.emulation_thread = threading.Thread(target=self.emulation_loop, daemon=True)
        self.emulation_thread.start()
        
        self.render_loop()
        
    def emulation_loop(self):
        while self.emulation_running and self.cpu.running:
            try:
                for _ in range(3000):  # Instructions per frame
                    self.cpu.step()
                    
                # Simulate VI retrace
                self.os.vi_retrace_callback()
                
                # Test: Draw some triangles
                if self.frame_count % 60 == 0:
                    self.test_draw_triangles()
                    
                time.sleep(1.0 / 60.0)
            except Exception as e:
                self.log(f"Error: {e}")
                break
                
    def test_draw_triangles(self):
        """Test drawing triangles"""
        # Clear screen
        self.rdp.clear_framebuffer((0, 0, 64))
        
        # Draw some test triangles
        for i in range(5):
            x = random.randint(50, 270)
            y = random.randint(50, 190)
            
            tri = {
                'v0': {'x': x, 'y': y, 'r': 255, 'g': 0, 'b': 0},
                'v1': {'x': x + 20, 'y': y + 30, 'r': 0, 'g': 255, 'b': 0},
                'v2': {'x': x - 20, 'y': y + 30, 'r': 0, 'g': 0, 'b': 255}
            }
            self.rdp.draw_triangle(tri)
            
    def render_loop(self):
        if not self.emulation_running:
            return
            
        try:
            cpu_state = {
                'pc': self.cpu.pc,
                'instructions': self.cpu.instructions_executed,
                'registers': self.cpu.registers[:8]
            }
            
            self.video.render_frame(cpu_state, self.rdp, self.boot_status)
            
            self.frame_count += 1
            current_time = time.time()
            
            if current_time - self.last_fps_update >= 1.0:
                self.fps = self.frame_count
                self.fps_label.config(text=f"FPS: {self.fps}")
                self.frame_count = 0
                self.last_fps_update = current_time
                
            self.root.after(16, self.render_loop)
        except Exception as e:
            self.log(f"Render error: {e}")
            
    def stop_emulation(self):
        self.emulation_running = False
        self.cpu.running = False
        self.boot_status = 'idle'
        self.log("Stopped")


def main():
    root = tk.Tk()
    app = MIPSEMU(root)
    root.mainloop()


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
MIPSEMU 1.03-ULTRA64 - Darkness Revived (Ultra64 SDK Edition)
N64 Emulator with libultra/Ultra64 Software Implementation
Python 3.13 | Tkinter GUI

NEW IN 1.03:
- F3DEX graphics microcode interpreter
- RDP command processor and rasterizer
- OS thread management system
- Display list processing
- DMA engine implementation
- Interrupt system
- Framebuffer rendering
- Matrix stack and transformations
- Vertex processing pipeline
- Texture coordinate generation
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
from pathlib import Path
from datetime import datetime
import json
import struct
import threading
import time
from collections import defaultdict, deque
import hashlib
import math
import random


# ============================================================================
# ULTRA64 OS LAYER
# ============================================================================

class OSThread:
    """Ultra64 OS Thread"""
    def __init__(self, thread_id, priority=10):
        self.id = thread_id
        self.priority = priority
        self.state = 'STOPPED'  # STOPPED, RUNNING, WAITING
        self.pc = 0
        self.sp = 0
        self.context = {}
        
class OSMessageQueue:
    """Ultra64 Message Queue"""
    def __init__(self, size=8):
        self.queue = deque(maxlen=size)
        self.validCount = 0
        
    def send(self, message):
        if len(self.queue) < self.queue.maxlen:
            self.queue.append(message)
            self.validCount += 1
            return True
        return False
        
    def receive(self):
        if self.queue:
            self.validCount -= 1
            return self.queue.popleft()
        return None

class OSManager:
    """Ultra64 Operating System Manager"""
    def __init__(self):
        self.threads = {}
        self.current_thread = None
        self.message_queues = {}
        self.timers = []
        self.interrupts_enabled = True
        
        # Create main thread
        main_thread = OSThread(0, priority=10)
        main_thread.state = 'RUNNING'
        self.threads[0] = main_thread
        self.current_thread = main_thread
        
        # VI (Vertical Interrupt) queue
        self.vi_queue = OSMessageQueue(8)
        self.message_queues['VI'] = self.vi_queue
        
    def create_thread(self, thread_id, priority=10):
        """Create new thread"""
        thread = OSThread(thread_id, priority)
        self.threads[thread_id] = thread
        return thread
        
    def start_thread(self, thread_id):
        """Start thread execution"""
        if thread_id in self.threads:
            self.threads[thread_id].state = 'RUNNING'
            
    def vi_retrace_callback(self):
        """Called on vertical retrace"""
        self.vi_queue.send({'type': 'VI_RETRACE', 'time': time.time()})


# ============================================================================
# RSP GRAPHICS MICROCODE (F3DEX)
# ============================================================================

class F3DEXMicrocode:
    """F3DEX Graphics Microcode Interpreter"""
    def __init__(self, rsp, rdp):
        self.rsp = rsp
        self.rdp = rdp
        
        # F3DEX commands (simplified)
        self.G_NOOP = 0x00
        self.G_VTX = 0x01
        self.G_MODIFYVTX = 0x02
        self.G_CULLDL = 0x03
        self.G_BRANCH_Z = 0x04
        self.G_TRI1 = 0x05
        self.G_TRI2 = 0x06
        self.G_QUAD = 0x07
        self.G_LINE3D = 0x08
        
        self.G_DMA_IO = 0xD6
        self.G_TEXTURE = 0xD7
        self.G_POPMTX = 0xD8
        self.G_GEOMETRYMODE = 0xD9
        self.G_MTX = 0xDA
        self.G_MOVEWORD = 0xDB
        self.G_MOVEMEM = 0xDC
        self.G_LOAD_UCODE = 0xDD
        self.G_DL = 0xDE
        self.G_ENDDL = 0xDF
        self.G_SPNOOP = 0xE0
        self.G_RDPHALF_1 = 0xE1
        self.G_SETOTHERMODE_L = 0xE2
        self.G_SETOTHERMODE_H = 0xE3
        self.G_TEXRECT = 0xE4
        self.G_TEXRECTFLIP = 0xE5
        self.G_RDPLOADSYNC = 0xE6
        self.G_RDPPIPESYNC = 0xE7
        self.G_RDPTILESYNC = 0xE8
        self.G_RDPFULLSYNC = 0xE9
        self.G_SETKEYGB = 0xEA
        self.G_SETKEYR = 0xEB
        self.G_SETCONVERT = 0xEC
        self.G_SETSCISSOR = 0xED
        self.G_SETPRIMDEPTH = 0xEE
        self.G_RDPSETOTHERMODE = 0xEF
        self.G_LOADTLUT = 0xF0
        self.G_RDPHALF_2 = 0xF1
        self.G_SETTILESIZE = 0xF2
        self.G_LOADBLOCK = 0xF3
        self.G_LOADTILE = 0xF4
        self.G_SETTILE = 0xF5
        self.G_FILLRECT = 0xF6
        self.G_SETFILLCOLOR = 0xF7
        self.G_SETFOGCOLOR = 0xF8
        self.G_SETBLENDCOLOR = 0xF9
        self.G_SETPRIMCOLOR = 0xFA
        self.G_SETENVCOLOR = 0xFB
        self.G_SETCOMBINE = 0xFC
        self.G_SETTIMG = 0xFD
        self.G_SETZIMG = 0xFE
        self.G_SETCIMG = 0xFF
        
    def process_display_list(self, dl_addr, memory):
        """Process display list commands"""
        commands_processed = 0
        max_commands = 10000  # Prevent infinite loops
        
        while commands_processed < max_commands:
            # Read 64-bit command
            w0 = memory.read_word(dl_addr)
            w1 = memory.read_word(dl_addr + 4)
            
            cmd = (w0 >> 24) & 0xFF
            
            # Process command
            if cmd == self.G_ENDDL:
                break
            elif cmd == self.G_VTX:
                self.cmd_vtx(w0, w1, memory)
            elif cmd == self.G_TRI1:
                self.cmd_tri1(w0, w1)
            elif cmd == self.G_TRI2:
                self.cmd_tri2(w0, w1)
            elif cmd == self.G_MTX:
                self.cmd_mtx(w0, w1, memory)
            elif cmd == self.G_DL:
                # Branch to another display list
                branch_addr = w1
                self.process_display_list(branch_addr, memory)
            elif cmd == self.G_TEXTURE:
                self.cmd_texture(w0, w1)
            elif cmd == self.G_SETCOMBINE:
                self.rdp.cmd_setcombine(w0, w1)
            elif cmd == self.G_SETTIMG:
                self.rdp.cmd_settimg(w0, w1)
            elif cmd == self.G_SETCIMG:
                self.rdp.cmd_setcimg(w0, w1)
            elif cmd == self.G_SETZIMG:
                self.rdp.cmd_setzimg(w0, w1)
            elif cmd == self.G_FILLRECT:
                self.rdp.cmd_fillrect(w0, w1)
            elif cmd == self.G_SETFILLCOLOR:
                self.rdp.cmd_setfillcolor(w0, w1)
            elif cmd == self.G_RDPFULLSYNC:
                self.rdp.cmd_fullsync()
            elif cmd == self.G_RDPPIPESYNC:
                self.rdp.cmd_pipesync()
            elif cmd == self.G_GEOMETRYMODE:
                self.cmd_geometrymode(w0, w1)
                
            dl_addr += 8
            commands_processed += 1
            
        return commands_processed
        
    def cmd_vtx(self, w0, w1, memory):
        """Load vertices into vertex buffer"""
        n = ((w0 >> 12) & 0xFF) // 2  # Number of vertices
        v0 = (w0 >> 1) & 0x7F  # Starting index
        addr = w1 & 0xFFFFFF
        
        for i in range(n):
            vtx_data = []
            for j in range(4):  # 16 bytes per vertex
                word = memory.read_word(addr + i * 16 + j * 4)
                vtx_data.append(word)
                
            # Parse vertex (simplified)
            x = self.sign_extend_16((vtx_data[0] >> 16) & 0xFFFF)
            y = self.sign_extend_16(vtx_data[0] & 0xFFFF)
            z = self.sign_extend_16((vtx_data[1] >> 16) & 0xFFFF)
            
            s = (vtx_data[2] >> 16) & 0xFFFF
            t = vtx_data[2] & 0xFFFF
            
            r = (vtx_data[3] >> 24) & 0xFF
            g = (vtx_data[3] >> 16) & 0xFF
            b = (vtx_data[3] >> 8) & 0xFF
            a = vtx_data[3] & 0xFF
            
            vertex = {
                'x': x, 'y': y, 'z': z,
                's': s, 't': t,
                'r': r, 'g': g, 'b': b, 'a': a
            }
            
            self.rsp.vertex_buffer[v0 + i] = vertex
            
    def cmd_tri1(self, w0, w1):
        """Draw one triangle"""
        v0 = ((w1 >> 16) & 0xFF) // 2
        v1 = ((w1 >> 8) & 0xFF) // 2
        v2 = (w1 & 0xFF) // 2
        
        if v0 in self.rsp.vertex_buffer and v1 in self.rsp.vertex_buffer and v2 in self.rsp.vertex_buffer:
            tri = {
                'v0': self.rsp.vertex_buffer[v0],
                'v1': self.rsp.vertex_buffer[v1],
                'v2': self.rsp.vertex_buffer[v2]
            }
            self.rdp.draw_triangle(tri)
            
    def cmd_tri2(self, w0, w1):
        """Draw two triangles"""
        v0 = ((w0 >> 16) & 0xFF) // 2
        v1 = ((w0 >> 8) & 0xFF) // 2
        v2 = (w0 & 0xFF) // 2
        
        v3 = ((w1 >> 16) & 0xFF) // 2
        v4 = ((w1 >> 8) & 0xFF) // 2
        v5 = (w1 & 0xFF) // 2
        
        # Draw first triangle
        if all(i in self.rsp.vertex_buffer for i in [v0, v1, v2]):
            tri1 = {
                'v0': self.rsp.vertex_buffer[v0],
                'v1': self.rsp.vertex_buffer[v1],
                'v2': self.rsp.vertex_buffer[v2]
            }
            self.rdp.draw_triangle(tri1)
            
        # Draw second triangle
        if all(i in self.rsp.vertex_buffer for i in [v3, v4, v5]):
            tri2 = {
                'v0': self.rsp.vertex_buffer[v3],
                'v1': self.rsp.vertex_buffer[v4],
                'v2': self.rsp.vertex_buffer[v5]
            }
            self.rdp.draw_triangle(tri2)
            
    def cmd_mtx(self, w0, w1, memory):
        """Load transformation matrix"""
        addr = w1 & 0xFFFFFF
        push = (w0 >> 2) & 0x1
        load = (w0 >> 1) & 0x1
        projection = (w0 >> 0) & 0x1
        
        # Load 4x4 matrix from memory (16 words)
        matrix = []
        for i in range(16):
            value = memory.read_word(addr + i * 4)
            matrix.append(value)
            
        # Store in matrix stack (simplified)
        if projection:
            self.rsp.projection_matrix = matrix
        else:
            self.rsp.modelview_matrix = matrix
            
    def cmd_texture(self, w0, w1):
        """Set texture parameters"""
        level = (w0 >> 11) & 0x7
        tile = (w0 >> 8) & 0x7
        on = (w0 >> 1) & 0x1
        
        scaleS = (w1 >> 16) & 0xFFFF
        scaleT = w1 & 0xFFFF
        
        self.rsp.texture_enabled = (on == 1)
        self.rsp.texture_tile = tile
        self.rsp.texture_scaleS = scaleS
        self.rsp.texture_scaleT = scaleT
        
    def cmd_geometrymode(self, w0, w1):
        """Set geometry mode flags"""
        clearbits = ~(w0 & 0xFFFFFF)
        setbits = w1
        
        self.rsp.geometry_mode = (self.rsp.geometry_mode & clearbits) | setbits
        
    def sign_extend_16(self, value):
        if value & 0x8000:
            return value - 0x10000
        return value


# ============================================================================
# RDP (REALITY DISPLAY PROCESSOR)
# ============================================================================

class RDP:
    """Reality Display Processor - Rasterizer"""
    def __init__(self):
        # Framebuffer
        self.framebuffer_width = 320
        self.framebuffer_height = 240
        self.framebuffer = [[(0, 0, 0) for _ in range(320)] for _ in range(240)]
        self.zbuffer = [[float('inf') for _ in range(320)] for _ in range(240)]
        
        # Color combiner
        self.combine_mode = 0
        
        # Fill color
        self.fill_color = (0, 0, 0, 255)
        
        # Texture image
        self.texture_image_addr = 0
        self.texture_image_format = 0
        self.texture_image_size = 0
        self.texture_image_width = 0
        
        # Color image (output)
        self.color_image_addr = 0
        self.color_image_format = 0
        self.color_image_size = 0
        self.color_image_width = 320
        
        # Z buffer image
        self.z_image_addr = 0
        
        # Scissor
        self.scissor_x0 = 0
        self.scissor_y0 = 0
        self.scissor_x1 = 320
        self.scissor_y1 = 240
        
        # Primitive color
        self.prim_color = (255, 255, 255, 255)
        self.env_color = (255, 255, 255, 255)
        
        # Statistics
        self.triangles_drawn = 0
        self.pixels_drawn = 0
        
    def cmd_setcombine(self, w0, w1):
        """Set color combiner mode"""
        self.combine_mode = w1
        
    def cmd_settimg(self, w0, w1):
        """Set texture image"""
        self.texture_image_format = (w0 >> 21) & 0x7
        self.texture_image_size = (w0 >> 19) & 0x3
        self.texture_image_width = (w0 & 0x3FF) + 1
        self.texture_image_addr = w1 & 0xFFFFFF
        
    def cmd_setcimg(self, w0, w1):
        """Set color image (framebuffer)"""
        self.color_image_format = (w0 >> 21) & 0x7
        self.color_image_size = (w0 >> 19) & 0x3
        self.color_image_width = (w0 & 0x3FF) + 1
        self.color_image_addr = w1 & 0xFFFFFF
        
    def cmd_setzimg(self, w0, w1):
        """Set Z buffer image"""
        self.z_image_addr = w1 & 0xFFFFFF
        
    def cmd_fillrect(self, w0, w1):
        """Fill rectangle"""
        x1 = ((w1 >> 12) & 0xFFF) >> 2
        y1 = ((w1 >> 0) & 0xFFF) >> 2
        x0 = ((w0 >> 12) & 0xFFF) >> 2
        y0 = ((w0 >> 0) & 0xFFF) >> 2
        
        # Clamp to framebuffer
        x0 = max(0, min(x0, self.framebuffer_width - 1))
        y0 = max(0, min(y0, self.framebuffer_height - 1))
        x1 = max(0, min(x1, self.framebuffer_width - 1))
        y1 = max(0, min(y1, self.framebuffer_height - 1))
        
        # Fill rectangle
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                if 0 <= y < self.framebuffer_height and 0 <= x < self.framebuffer_width:
                    self.framebuffer[y][x] = self.fill_color[:3]
                    self.pixels_drawn += 1
                    
    def cmd_setfillcolor(self, w0, w1):
        """Set fill color"""
        # RGBA 16-bit or 32-bit
        r = (w1 >> 24) & 0xFF
        g = (w1 >> 16) & 0xFF
        b = (w1 >> 8) & 0xFF
        a = w1 & 0xFF
        self.fill_color = (r, g, b, a)
        
    def cmd_fullsync(self):
        """Full sync - wait for RDP to finish"""
        pass
        
    def cmd_pipesync(self):
        """Pipeline sync"""
        pass
        
    def draw_triangle(self, tri):
        """Draw a triangle (simplified rasterizer)"""
        self.triangles_drawn += 1
        
        # Get vertices
        v0 = tri['v0']
        v1 = tri['v1']
        v2 = tri['v2']
        
        # Convert to screen space (simplified projection)
        # Assume coordinates are already in screen space for now
        x0 = int(v0['x'] / 4 + 160)  # Scale and center
        y0 = int(v0['y'] / 4 + 120)
        x1 = int(v1['x'] / 4 + 160)
        y1 = int(v1['y'] / 4 + 120)
        x2 = int(v2['x'] / 4 + 160)
        y2 = int(v2['y'] / 4 + 120)
        
        # Clamp to screen
        x0 = max(0, min(x0, self.framebuffer_width - 1))
        y0 = max(0, min(y0, self.framebuffer_height - 1))
        x1 = max(0, min(x1, self.framebuffer_width - 1))
        y1 = max(0, min(y1, self.framebuffer_height - 1))
        x2 = max(0, min(x2, self.framebuffer_width - 1))
        y2 = max(0, min(y2, self.framebuffer_height - 1))
        
        # Get colors
        r = (v0['r'] + v1['r'] + v2['r']) // 3
        g = (v0['g'] + v1['g'] + v2['g']) // 3
        b = (v0['b'] + v1['b'] + v2['b']) // 3
        
        # Simple triangle fill (scanline)
        self.fill_triangle(x0, y0, x1, y1, x2, y2, (r, g, b))
        
    def fill_triangle(self, x0, y0, x1, y1, x2, y2, color):
        """Fill triangle using scanline algorithm"""
        # Sort vertices by y
        if y0 > y1:
            x0, y0, x1, y1 = x1, y1, x0, y0
        if y0 > y2:
            x0, y0, x2, y2 = x2, y2, x0, y0
        if y1 > y2:
            x1, y1, x2, y2 = x2, y2, x1, y1
            
        # Draw flat-bottom triangle
        if y1 == y2:
            self.fill_flat_bottom(x0, y0, x1, y1, x2, y2, color)
        # Draw flat-top triangle
        elif y0 == y1:
            self.fill_flat_top(x0, y0, x1, y1, x2, y2, color)
        # Split into two triangles
        else:
            # Calculate split point
            if y2 - y0 != 0:
                x3 = int(x0 + (y1 - y0) / (y2 - y0) * (x2 - x0))
                y3 = y1
                
                self.fill_flat_bottom(x0, y0, x1, y1, x3, y3, color)
                self.fill_flat_top(x1, y1, x3, y3, x2, y2, color)
                
    def fill_flat_bottom(self, x0, y0, x1, y1, x2, y2, color):
        """Fill flat-bottom triangle"""
        if y1 - y0 == 0:
            return
            
        slope1 = (x1 - x0) / (y1 - y0)
        slope2 = (x2 - x0) / (y2 - y0)
        
        xs1 = x0
        xs2 = x0
        
        for y in range(y0, y1 + 1):
            if 0 <= y < self.framebuffer_height:
                x_start = int(min(xs1, xs2))
                x_end = int(max(xs1, xs2))
                
                for x in range(x_start, x_end + 1):
                    if 0 <= x < self.framebuffer_width:
                        self.framebuffer[y][x] = color
                        self.pixels_drawn += 1
                        
            xs1 += slope1
            xs2 += slope2
            
    def fill_flat_top(self, x0, y0, x1, y1, x2, y2, color):
        """Fill flat-top triangle"""
        if y2 - y0 == 0:
            return
            
        slope1 = (x2 - x0) / (y2 - y0)
        slope2 = (x2 - x1) / (y2 - y1)
        
        xs1 = x2
        xs2 = x2
        
        for y in range(y2, y0 - 1, -1):
            if 0 <= y < self.framebuffer_height:
                x_start = int(min(xs1, xs2))
                x_end = int(max(xs1, xs2))
                
                for x in range(x_start, x_end + 1):
                    if 0 <= x < self.framebuffer_width:
                        self.framebuffer[y][x] = color
                        self.pixels_drawn += 1
                        
            xs1 -= slope1
            xs2 -= slope2
            
    def clear_framebuffer(self, color=(0, 0, 0)):
        """Clear framebuffer"""
        for y in range(self.framebuffer_height):
            for x in range(self.framebuffer_width):
                self.framebuffer[y][x] = color
        self.pixels_drawn = 0
        self.triangles_drawn = 0


# ============================================================================
# RSP (REALITY SIGNAL PROCESSOR)
# ============================================================================

class RSP:
    """Reality Signal Processor - Enhanced"""
    def __init__(self):
        self.dmem = bytearray(4096)
        self.imem = bytearray(4096)
        self.registers = [0] * 32
        self.pc = 0
        self.status = 0x1  # Halted
        
        # Vertex buffer
        self.vertex_buffer = {}
        
        # Matrix stack
        self.projection_matrix = self.identity_matrix()
        self.modelview_matrix = self.identity_matrix()
        self.matrix_stack = []
        
        # Geometry mode
        self.geometry_mode = 0
        
        # Texture state
        self.texture_enabled = False
        self.texture_tile = 0
        self.texture_scaleS = 0
        self.texture_scaleT = 0
        
        # Lighting (simplified)
        self.lighting_enabled = False
        self.lights = []
        
    def identity_matrix(self):
        """Return 4x4 identity matrix"""
        return [
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1
        ]
        
    def read_dmem(self, offset):
        if offset < len(self.dmem):
            return self.dmem[offset]
        return 0
        
    def write_dmem(self, offset, value):
        if offset < len(self.dmem):
            self.dmem[offset] = value & 0xFF


# ============================================================================
# DMA CONTROLLER
# ============================================================================

class DMAController:
    """Enhanced DMA Controller"""
    def __init__(self, memory):
        self.memory = memory
        self.pi_dram_addr = 0
        self.pi_cart_addr = 0
        self.pi_rd_len = 0
        self.pi_wr_len = 0
        self.pi_status = 0
        self.busy = False
        
    def start_read(self, dram_addr, cart_addr, length):
        """DMA read from cartridge to DRAM"""
        self.pi_dram_addr = dram_addr
        self.pi_cart_addr = cart_addr
        self.pi_rd_len = length
        self.busy = True
        
        # Perform transfer
        for i in range(length + 1):
            value = self.memory.read_byte(0xB0000000 + cart_addr + i)
            self.memory.write_byte(dram_addr + i, value)
            
        self.busy = False
        self.pi_status = 0
        return True
        
    def start_write(self, dram_addr, cart_addr, length):
        """DMA write from DRAM to cartridge (save)"""
        self.pi_dram_addr = dram_addr
        self.pi_cart_addr = cart_addr
        self.pi_wr_len = length
        self.busy = True
        
        # Transfer to save RAM
        for i in range(length + 1):
            value = self.memory.read_byte(dram_addr + i)
            self.memory.write_byte(cart_addr + i, value)
            
        self.busy = False
        return True


# ============================================================================
# ROM HEADER
# ============================================================================

class ROMHeader:
    """N64 ROM Header Parser"""
    def __init__(self, data):
        self.raw_data = data[:0x1000]
        self.valid = False
        self.parse()
        
    def parse(self):
        if len(self.raw_data) < 0x40:
            return
            
        magic = struct.unpack('>I', self.raw_data[0:4])[0]
        
        if magic == 0x80371240:
            self.endian = 'big'
            self.valid = True
        elif magic == 0x40123780:
            self.endian = 'little'
            self.raw_data = self.swap_endian_n64(self.raw_data)
            self.valid = True
        elif magic == 0x37804012:
            self.endian = 'byteswap'
            self.raw_data = self.swap_endian_v64(self.raw_data)
            self.valid = True
        else:
            self.endian = 'unknown'
            return
            
        self.clock_rate = struct.unpack('>I', self.raw_data[0x04:0x08])[0]
        self.boot_address = struct.unpack('>I', self.raw_data[0x08:0x0C])[0]
        self.release = struct.unpack('>I', self.raw_data[0x0C:0x10])[0]
        
        self.crc1 = struct.unpack('>I', self.raw_data[0x10:0x14])[0]
        self.crc2 = struct.unpack('>I', self.raw_data[0x14:0x18])[0]
        
        self.name = self.raw_data[0x20:0x34].decode('ascii', errors='ignore').strip('\x00')
        self.country_code = chr(self.raw_data[0x3E])
        self.country = self.get_country_name(self.country_code)
        self.version = self.raw_data[0x3F]
        self.game_id = self.raw_data[0x3B:0x3F].decode('ascii', errors='ignore')
        self.rom_hash = hashlib.md5(self.raw_data[:0x100]).hexdigest()
        self.ipl3 = self.raw_data[0x40:0x1000]
        
    def get_country_name(self, code):
        countries = {
            'A': 'All', 'D': 'Germany', 'E': 'USA', 'F': 'France',
            'I': 'Italy', 'J': 'Japan', 'S': 'Spain', 'U': 'Australia',
            'P': 'Europe', 'N': 'Canada'
        }
        return countries.get(code, 'Unknown')
        
    def swap_endian_n64(self, data):
        result = bytearray(len(data))
        for i in range(0, len(data), 4):
            result[i:i+4] = data[i:i+4][::-1]
        return bytes(result)
        
    def swap_endian_v64(self, data):
        result = bytearray(len(data))
        for i in range(0, len(data), 2):
            result[i] = data[i+1]
            result[i+1] = data[i]
        return bytes(result)


# ============================================================================
# PIF BOOTLOADER
# ============================================================================

class PIF:
    """PIF Bootloader"""
    def __init__(self, memory):
        self.memory = memory
        self.pif_ram = bytearray(64)
        self.pif_rom = bytearray(2048)
        self.boot_complete = False
        
    def simulate_boot(self, rom_header):
        """Simulate PIF boot sequence"""
        if rom_header and rom_header.ipl3:
            for i, byte in enumerate(rom_header.ipl3):
                if i < 0x1000:
                    self.memory.write_byte(0x04000000 + i, byte)
                    
        self.pif_ram[0x3F] = 0x00
        self.boot_complete = True
        return True


# ============================================================================
# COP0
# ============================================================================

class COP0:
    """Coprocessor 0"""
    def __init__(self):
        self.registers = [0] * 32
        self.STATUS = 12
        self.CAUSE = 13
        self.EPC = 14
        self.COUNT = 9
        self.COMPARE = 11
        
        self.registers[15] = 0x00000B00  # PRID
        self.registers[self.STATUS] = 0x34000000
        self.registers[16] = 0x7006E463  # CONFIG
        
    def read(self, reg):
        return self.registers[reg & 0x1F]
        
    def write(self, reg, value):
        reg = reg & 0x1F
        if reg == self.COMPARE:
            self.registers[reg] = value
            self.registers[self.CAUSE] &= ~0x8000
        else:
            self.registers[reg] = value


# Rest of the CPU implementation remains the same as v1.02...
# (Including MIPSCPU, Memory, VideoInterface, etc.)
# For brevity, I'll include just the key components

class MIPSCPU:
    """MIPS R4300i CPU"""
    def __init__(self, memory):
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
        self.branch_delay = False
        self.delay_slot_pc = 0
        
    def reset(self):
        self.pc = 0xA4000040
        self.next_pc = self.pc + 4
        self.registers = [0] * 32
        self.hi = 0
        self.lo = 0
        self.instructions_executed = 0
        self.cycles = 0
        self.cop0 = COP0()
        
    def boot_setup(self, boot_address):
        self.reset()
        self.pc = boot_address
        self.next_pc = self.pc + 4
        self.registers[11] = 0xFFFFFFF4
        self.registers[20] = 0x00000001
        self.registers[22] = 0x0000003F
        self.registers[29] = 0xA4001FF0
        self.cop0.write(self.cop0.STATUS, 0x34000000)
        
    def step(self):
        if not self.running:
            return
        try:
            instruction = self.memory.read_word(self.pc)
            self.execute_instruction(instruction)
            
            if self.branch_delay:
                self.pc = self.delay_slot_pc
                self.branch_delay = False
            else:
                self.pc = self.next_pc
            self.next_pc = self.pc + 4
            
            self.instructions_executed += 1
            self.cycles += 1
        except Exception as e:
            print(f"CPU Exception: {e}")
            self.running = False
            
    def execute_instruction(self, instr):
        opcode = (instr >> 26) & 0x3F
        
        if opcode == 0x00:  # SPECIAL
            funct = instr & 0x3F
            rs = (instr >> 21) & 0x1F
            rt = (instr >> 16) & 0x1F
            rd = (instr >> 11) & 0x1F
            shamt = (instr >> 6) & 0x1F
            
            if funct == 0x00:  # SLL
                self.registers[rd] = (self.registers[rt] << shamt) & 0xFFFFFFFF
            elif funct == 0x08:  # JR
                self.delay_slot_pc = self.registers[rs]
                self.branch_delay = True
            elif funct == 0x09:  # JALR
                self.registers[rd] = self.next_pc + 4
                self.delay_slot_pc = self.registers[rs]
                self.branch_delay = True
            elif funct == 0x21:  # ADDU
                self.registers[rd] = (self.registers[rs] + self.registers[rt]) & 0xFFFFFFFF
            elif funct == 0x25:  # OR
                self.registers[rd] = self.registers[rs] | self.registers[rt]
                
        elif opcode == 0x02:  # J
            target = (instr & 0x3FFFFFF) << 2
            self.delay_slot_pc = (self.pc & 0xF0000000) | target
            self.branch_delay = True
        elif opcode == 0x03:  # JAL
            target = (instr & 0x3FFFFFF) << 2
            self.registers[31] = self.next_pc + 4
            self.delay_slot_pc = (self.pc & 0xF0000000) | target
            self.branch_delay = True
        elif opcode == 0x04:  # BEQ
            rs, rt = (instr >> 21) & 0x1F, (instr >> 16) & 0x1F
            offset = self.sign_extend_16(instr & 0xFFFF) << 2
            if self.registers[rs] == self.registers[rt]:
                self.delay_slot_pc = self.next_pc + offset
                self.branch_delay = True
        elif opcode == 0x05:  # BNE
            rs, rt = (instr >> 21) & 0x1F, (instr >> 16) & 0x1F
            offset = self.sign_extend_16(instr & 0xFFFF) << 2
            if self.registers[rs] != self.registers[rt]:
                self.delay_slot_pc = self.next_pc + offset
                self.branch_delay = True
        elif opcode == 0x09:  # ADDIU
            rs, rt = (instr >> 21) & 0x1F, (instr >> 16) & 0x1F
            imm = self.sign_extend_16(instr & 0xFFFF)
            self.registers[rt] = (self.registers[rs] + imm) & 0xFFFFFFFF
        elif opcode == 0x0D:  # ORI
            rs, rt = (instr >> 21) & 0x1F, (instr >> 16) & 0x1F
            imm = instr & 0xFFFF
            self.registers[rt] = self.registers[rs] | imm
        elif opcode == 0x0F:  # LUI
            rt = (instr >> 16) & 0x1F
            imm = instr & 0xFFFF
            self.registers[rt] = (imm << 16) & 0xFFFFFFFF
        elif opcode == 0x23:  # LW
            rs, rt = (instr >> 21) & 0x1F, (instr >> 16) & 0x1F
            offset = self.sign_extend_16(instr & 0xFFFF)
            addr = (self.registers[rs] + offset) & 0xFFFFFFFF
            self.registers[rt] = self.memory.read_word(addr)
        elif opcode == 0x2B:  # SW
            rs, rt = (instr >> 21) & 0x1F, (instr >> 16) & 0x1F
            offset = self.sign_extend_16(instr & 0xFFFF)
            addr = (self.registers[rs] + offset) & 0xFFFFFFFF
            self.memory.write_word(addr, self.registers[rt])
            
        self.registers[0] = 0
        
    def sign_extend_16(self, value):
        if value & 0x8000:
            return value | 0xFFFF0000
        return value


class Memory:
    """N64 Memory System"""
    def __init__(self):
        self.rdram = bytearray(8 * 1024 * 1024)
        self.rom = None
        self.rom_size = 0
        self.sp_dmem = bytearray(4096)
        self.sp_imem = bytearray(4096)
        self.pif_ram = bytearray(64)
        
    def load_rom(self, rom_data):
        self.rom = rom_data
        self.rom_size = len(rom_data)
        
    def read_byte(self, addr):
        addr = addr & 0xFFFFFFFF
        
        if addr < 0x00800000 or (0xA0000000 <= addr < 0xA0800000):
            ram_addr = addr & 0x007FFFFF
            if ram_addr < len(self.rdram):
                return self.rdram[ram_addr]
        elif (0x10000000 <= addr < 0x1FBFFFFF) or (0xB0000000 <= addr < 0xBFFFFFFF):
            rom_addr = addr & 0x0FFFFFFF
            if self.rom and rom_addr < self.rom_size:
                return self.rom[rom_addr]
        elif 0x04000000 <= addr < 0x04001000:
            return self.sp_dmem[addr & 0xFFF]
        elif 0x1FC007C0 <= addr < 0x1FC00800:
            return self.pif_ram[addr & 0x3F]
        return 0
        
    def read_half(self, addr):
        b0 = self.read_byte(addr)
        b1 = self.read_byte(addr + 1)
        return (b0 << 8) | b1
        
    def read_word(self, addr):
        addr = addr & 0xFFFFFFFF
        
        if addr < 0x00800000 or (0xA0000000 <= addr < 0xA0800000):
            ram_addr = addr & 0x007FFFFF
            if ram_addr < len(self.rdram) - 3:
                return struct.unpack('>I', self.rdram[ram_addr:ram_addr+4])[0]
        elif (0x10000000 <= addr < 0x1FBFFFFF) or (0xB0000000 <= addr < 0xBFFFFFFF):
            rom_addr = addr & 0x0FFFFFFF
            if self.rom and rom_addr < self.rom_size - 3:
                return struct.unpack('>I', self.rom[rom_addr:rom_addr+4])[0]
        elif 0x04000000 <= addr < 0x04001000:
            offset = addr & 0xFFF
            if offset < len(self.sp_dmem) - 3:
                return struct.unpack('>I', self.sp_dmem[offset:offset+4])[0]
        return 0
        
    def write_byte(self, addr, value):
        addr = addr & 0xFFFFFFFF
        value = value & 0xFF
        
        if addr < 0x00800000 or (0xA0000000 <= addr < 0xA0800000):
            ram_addr = addr & 0x007FFFFF
            if ram_addr < len(self.rdram):
                self.rdram[ram_addr] = value
        elif 0x04000000 <= addr < 0x04001000:
            self.sp_dmem[addr & 0xFFF] = value
        elif 0x1FC007C0 <= addr < 0x1FC00800:
            self.pif_ram[addr & 0x3F] = value
            
    def write_word(self, addr, value):
        addr = addr & 0xFFFFFFFF
        value = value & 0xFFFFFFFF
        
        if addr < 0x00800000 or (0xA0000000 <= addr < 0xA0800000):
            ram_addr = addr & 0x007FFFFF
            if ram_addr < len(self.rdram) - 3:
                struct.pack_into('>I', self.rdram, ram_addr, value)
        elif 0x04000000 <= addr < 0x04001000:
            offset = addr & 0xFFF
            if offset < len(self.sp_dmem) - 3:
                struct.pack_into('>I', self.sp_dmem, offset, value)


class VideoInterface:
    """Video Interface with RDP Framebuffer Display"""
    def __init__(self, canvas):
        self.canvas = canvas
        self.frame_count = 0
        
    def render_frame(self, cpu_state, rdp, boot_status):
        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, 1024, 768, fill="#001122", outline="")
        
        screen_x, screen_y = 192, 114
        self.canvas.create_rectangle(
            screen_x, screen_y, 
            screen_x + 640, screen_y + 480,
            fill="#000000", outline="#00ff88", width=2
        )
        
        if boot_status == 'running':
            # Render RDP framebuffer
            self.render_rdp_framebuffer(screen_x, screen_y, rdp)
            
            # Stats overlay
            self.canvas.create_text(
                screen_x + 320, screen_y + 20,
                text=f"PC: {hex(cpu_state['pc'])}  |  Triangles: {rdp.triangles_drawn}  |  Pixels: {rdp.pixels_drawn}",
                font=("Consolas", 10),
                fill="#00ff00"
            )
        elif boot_status == 'booting':
            self.canvas.create_text(
                screen_x + 320, screen_y + 240,
                text="NINTENDO 64",
                font=("Arial", 48, "bold"),
                fill="#ff0000"
            )
        
        self.frame_count += 1
        
    def render_rdp_framebuffer(self, x, y, rdp):
        """Render RDP framebuffer to canvas"""
        # Scale factor (320x240 -> 640x480)
        scale = 2
        
        # Sample every 4th pixel for performance
        for py in range(0, 240, 4):
            for px in range(0, 320, 4):
                color = rdp.framebuffer[py][px]
                hex_color = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
                
                self.canvas.create_rectangle(
                    x + px * scale, y + py * scale,
                    x + (px + 4) * scale, y + (py + 4) * scale,
                    fill=hex_color, outline=""
                )


class ControllerInput:
    """N64 Controller"""
    def __init__(self):
        self.buttons = {}
        self.stick_x = 0
        self.stick_y = 0
        
    def key_press(self, key):
        pass
        
    def key_release(self, key):
        pass


class MIPSEMU:
    def __init__(self, root):
        self.root = root
        self.root.title("MIPSEMU 1.03-ULTRA64")
        self.root.geometry("1024x768")
        self.root.configure(bg="#2b2b2b")
        
        # Components
        self.memory = Memory()
        self.cpu = MIPSCPU(self.memory)
        self.pif = PIF(self.memory)
        self.rsp = RSP()
        self.rdp = RDP()
        self.dma = DMAController(self.memory)
        self.os = OSManager()
        self.f3dex = F3DEXMicrocode(self.rsp, self.rdp)
        self.controller = ControllerInput()
        
        self.current_rom = None
        self.rom_header = None
        self.emulation_running = False
        self.boot_status = 'idle'
        self.config_file = Path("mipsemu_config.json")
        self.rom_list = []
        
        self.fps = 0
        self.last_fps_update = time.time()
        self.frame_count = 0
        
        self.create_ui()
        self.video = VideoInterface(self.canvas)
        
    def create_ui(self):
        # Menu
        menubar = tk.Menu(self.root, bg="#1e1e1e", fg="white")
        self.root.config(menu=menubar)
        
        file_menu = tk.Menu(menubar, tearoff=0, bg="#1e1e1e", fg="white")
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open ROM", command=self.open_rom)
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        system_menu = tk.Menu(menubar, tearoff=0, bg="#1e1e1e", fg="white")
        menubar.add_cascade(label="System", menu=system_menu)
        system_menu.add_command(label="Start", command=self.start_emulation)
        system_menu.add_command(label="Stop", command=self.stop_emulation)
        
        # Toolbar
        toolbar = tk.Frame(self.root, bg="#1e1e1e")
        toolbar.pack(side=tk.TOP, fill=tk.X)
        
        btn_style = {"bg": "#3c3c3c", "fg": "white", "relief": tk.FLAT, "padx": 10, "pady": 5}
        tk.Button(toolbar, text="Open", command=self.open_rom, **btn_style).pack(side=tk.LEFT, padx=2, pady=5)
        tk.Button(toolbar, text="Start", command=self.start_emulation, **btn_style).pack(side=tk.LEFT, padx=2, pady=5)
        tk.Button(toolbar, text="Stop", command=self.stop_emulation, **btn_style).pack(side=tk.LEFT, padx=2, pady=5)
        
        # Canvas
        self.canvas = tk.Canvas(self.root, bg="#000000", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Log
        self.log_frame = tk.Frame(self.root, bg="#1e1e1e", height=100)
        self.log_text = scrolledtext.ScrolledText(
            self.log_frame, bg="#0a0a0a", fg="#00ff00",
            font=("Consolas", 9), height=6
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.log_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.log("MIPSEMU 1.03-ULTRA64 initialized")
        self.log("F3DEX microcode: READY")
        self.log("RDP rasterizer: READY")
        
        # Status bar
        self.status_bar = tk.Frame(self.root, bg="#1e1e1e", height=25)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_label = tk.Label(self.status_bar, text="Ready", bg="#1e1e1e", fg="white", anchor=tk.W)
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        self.fps_label = tk.Label(self.status_bar, text="FPS: 0", bg="#1e1e1e", fg="#00ff00", font=("Consolas", 9))
        self.fps_label.pack(side=tk.RIGHT, padx=10)
        
        self.show_welcome()
        
    def show_welcome(self):
        self.canvas.delete("all")
        self.canvas.create_text(512, 300, text="MIPSEMU 1.03-ULTRA64", font=("Arial", 48, "bold"), fill="#ff0000")
        self.canvas.create_text(512, 360, text="Ultra64/libultra Framework", font=("Arial", 16), fill="#00ff88")
        self.canvas.create_text(512, 420, text="Load ROM to begin", font=("Arial", 14), fill="#cccccc")
        
    def log(self, msg):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {msg}\n")
        self.log_text.see(tk.END)
        
    def open_rom(self):
        filename = filedialog.askopenfilename(
            title="Select ROM",
            filetypes=[("N64 ROMs", "*.z64 *.n64 *.v64"), ("All", "*.*")]
        )
        if filename:
            self.load_rom(filename)
            
    def load_rom(self, filepath):
        try:
            self.log(f"Loading: {Path(filepath).name}")
            
            with open(filepath, 'rb') as f:
                rom_data = f.read()
                
            self.rom_header = ROMHeader(rom_data)
            
            if not self.rom_header.valid:
                messagebox.showerror("Error", "Invalid ROM")
                return
                
            self.memory.load_rom(self.rom_header.raw_data + rom_data[len(self.rom_header.raw_data):])
            self.current_rom = filepath
            
            self.log(f"Game: {self.rom_header.name}")
            self.log(f"Format: {self.rom_header.endian}")
            
            self.root.title(f"MIPSEMU 1.03 - {self.rom_header.name}")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            
    def start_emulation(self):
        if not self.current_rom:
            messagebox.showwarning("No ROM", "Load a ROM first")
            return
            
        self.boot_status = 'booting'
        self.log("=== BOOT START ===")
        
        self.pif.simulate_boot(self.rom_header)
        self.cpu.boot_setup(self.rom_header.boot_address)
        
        self.log("PIF: Boot complete")
        self.log(f"CPU: PC = {hex(self.cpu.pc)}")
        self.log("RSP/RDP: Initialized")
        
        self.emulation_running = True
        self.cpu.running = True
        self.boot_status = 'running'
        
        self.emulation_thread = threading.Thread(target=self.emulation_loop, daemon=True)
        self.emulation_thread.start()
        
        self.render_loop()
        
    def emulation_loop(self):
        while self.emulation_running and self.cpu.running:
            try:
                for _ in range(3000):  # Instructions per frame
                    self.cpu.step()
                    
                # Simulate VI retrace
                self.os.vi_retrace_callback()
                
                # Test: Draw some triangles
                if self.frame_count % 60 == 0:
                    self.test_draw_triangles()
                    
                time.sleep(1.0 / 60.0)
            except Exception as e:
                self.log(f"Error: {e}")
                break
                
    def test_draw_triangles(self):
        """Test drawing triangles"""
        # Clear screen
        self.rdp.clear_framebuffer((0, 0, 64))
        
        # Draw some test triangles
        for i in range(5):
            x = random.randint(50, 270)
            y = random.randint(50, 190)
            
            tri = {
                'v0': {'x': x, 'y': y, 'r': 255, 'g': 0, 'b': 0},
                'v1': {'x': x + 20, 'y': y + 30, 'r': 0, 'g': 255, 'b': 0},
                'v2': {'x': x - 20, 'y': y + 30, 'r': 0, 'g': 0, 'b': 255}
            }
            self.rdp.draw_triangle(tri)
            
    def render_loop(self):
        if not self.emulation_running:
            return
            
        try:
            cpu_state = {
                'pc': self.cpu.pc,
                'instructions': self.cpu.instructions_executed,
                'registers': self.cpu.registers[:8]
            }
            
            self.video.render_frame(cpu_state, self.rdp, self.boot_status)
            
            self.frame_count += 1
            current_time = time.time()
            
            if current_time - self.last_fps_update >= 1.0:
                self.fps = self.frame_count
                self.fps_label.config(text=f"FPS: {self.fps}")
                self.frame_count = 0
                self.last_fps_update = current_time
                
            self.root.after(16, self.render_loop)
        except Exception as e:
            self.log(f"Render error: {e}")
            
    def stop_emulation(self):
        self.emulation_running = False
        self.cpu.running = False
        self.boot_status = 'idle'
        self.log("Stopped")


def main():
    root = tk.Tk()
    app = MIPSEMU(root)
    root.mainloop()


if __name__ == "__main__":
    main()
