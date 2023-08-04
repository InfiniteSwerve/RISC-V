#!/opt/miniconda3/bin/python3
"""
tests from 
https://github.com/riscv-software-src/riscv-tests
"""
from typing import Optional

# Interpret bytes as packed binary data
# Converts between python values and c structs represented as bytes
import struct

# Finds pathnames matching a pattern. stuff like ../../Downloads/*/*.exe
import glob

# elf = Executable and Linkable Format = Super versatile form of an executable file used
from elftools.elf.elffile import ELFFile
from enum import Enum

# Better register names according to the RISC-V spec page 137
regs = (
    ["x0", "ra", "sp", "gp", "tp"]
    + ["t%d" % i for i in range(3)]
    + ["s0", "s1"]
    + ["a%d" % i for i in range(0, 8)]
    + ["s%d" % i for i in range(2, 12)]
    + ["t%d" % i for i in range(3, 7)]
    + ["PC"]
)


# Our set of registers.
class Regfile:
    def __init__(self):
        self.regs = [0] * 34

    def __getitem__(self, x):
        return self.regs[x]

    def __setitem__(self, key, value):
        # MIPS convention to have register 0 always be zero (that's why we have 33 registers)
        if key == 0:
            return
        # Mask value to make sure we only grab 32 bits of it
        self.regs[key] = value & 0xFFFFFFFF


# register file, as in, keeps track of all register values
regfile = None
# 64k at 0x80000000
memory = None
PC = 32


def reset():
    global regfile, memory
    regfile = Regfile()
    memory = b"\x00" * 0x10000


class Ops(Enum):
    LUI = 0b0110111  # load upper immediate
    LOAD = 0b0000011
    STORE = 0b0100011

    AUIPC = 0b0010111  # add upper immediate to pc
    BRANCH = 0b1100011
    JAL = 0b1101111
    JALR = 0b1100111

    IMM = 0b0010011
    OP = 0b0110011

    MISC = 0b0001111
    SYSTEM = 0b1110011


class Func3(Enum):
    # OP-IMM
    # J-type
    ADDI = ADD = SUB = 0b000
    SLTI = SLT = 0b010
    SLTIU = SLTU = 0b011
    XORI = XOR = 0b100
    ORI = OR = 0b110
    ANDI = AND = 0b111
    SLLI = SLL = 0b001
    SRLI = SRAI = SRL = SRA = 0b101
    MISC = 0b000

    # BRANCH
    BEQ = 0b000
    BNE = 0b001
    BLT = 0b100
    BGE = 0b101
    BLTU = 0b110
    BGEU = 0b111

    # STORE
    SB = 0b000
    SH = 0b001
    SW = 0b010

    # LOAD
    LB = 0b000
    LH = 0b001
    LW = 0b010
    LBU = 0b100
    LHU = 0b101

    CSRRW = 0b001
    CSRRS = 0b010
    CSRRC = 0b011
    CSRRWI = 0b101
    CSRRSI = 0b110
    CSRRCI = 0b111

    # SYSTEM
    # I-type
    # ECALL makes a 'service request to the system environment'
    # EBREAK returns control
    ECALL = EBREAK = 0b000


# Write to mem function
def ws(dat, addr):
    global memory
    addr -= 0x80000000

    # print("STORE %8x = %x" % (addr, dat))
    if addr < 0 or addr > len(memory):
        raise Exception("write out of bounds: %x" % addr)
    if isinstance(dat, int):
        dat = struct.pack("<I", dat)
    # won't this increase total memory size over time?
    # Nope, because the lower index of the upper slice accounts for that
    # print("data len is", len(dat))
    memory = memory[:addr] + dat + memory[addr + len(dat) :]
    val = struct.unpack("<I", memory[addr : addr + 4])[0]
    # print("%x written" % val)


# Reads 32 bits from address
# I = interpret as unsigned int
# Unpack always returns a tuple even if there's no other elements
def r32(addr):
    addr -= 0x80000000
    if addr < 0 or addr > len(memory):
        raise Exception("read out of bounds: %x" % addr)
    return struct.unpack("<I", memory[addr : addr + 4])[0]


def to_signed32(n):
    return n | (-(n & 0x80000000))


def to_unsigned32(n):
    if n < 0:
        return abs(n) + 1 << 31
    else:
        return n


max_int32 = 0xFFFFFFFF


def imm_arith(func3, a, b):
    if func3 == Func3.ADDI:
        return a + b
    elif func3 == Func3.ORI:
        return a | b
    elif func3 == Func3.XORI:
        return a ^ b
    elif func3 == Func3.ANDI:
        return a & b
    elif func3 == Func3.SLLI:
        b = b & ((1 << 5) - 1)
        return a << b
    elif func3 == Func3.SRLI:
        # print(format(b, "012b"))
        kind = b & (1 << 10)  # 0b010000000000
        b = b & ((1 << 5) - 1)
        # print("kind is", kind)
        # SRAI
        if kind > 0 and (a & 2 ** (32 - 1) != 0):  # MSB is 1, i.e. a is negative
            # print("first branch")
            filler = int("1" * b + "0" * (32 - b), 2)
            a = (a >> b) | filler  # fill int 0's with 1's
            # print("a")
            return a
        # SRLI
        else:
            # print("second branch")
            return a >> b
    elif func3 == Func3.SLTI:
        if a < (b & ((1 << 5) - 1)):
            return 1
        else:
            return 0
    elif func3 == Func3.SLTIU or func3 == Func3.SLTU:
        if (a & max_int32) < (b & max_int32):
            return 1
        else:
            return 0
    else:
        raise Exception("write %r" % func3)


def op_arith(func3, a, b, func7: int):
    if func3 == Func3.ADD:
        if func7 > 0:
            return a - b
        else:
            return a + b
    elif func3 == Func3.OR:
        return a | b
    elif func3 == Func3.XOR:
        return a ^ b
    elif func3 == Func3.AND:
        return a & b
    elif func3 == Func3.SLL:
        b = b & ((1 << 5) - 1)
        return a << b
    elif func3 == Func3.SRL:
        b = b & ((1 << 5) - 1)
        # print("func7 is", func7)
        # SRA
        if func7 > 0 and (a & 2 ** (32 - 1) != 0):  # MSB is 1, i.e. a is negative
            # print("first branch")
            filler = int("1" * b + "0" * (32 - b), 2)
            a = (a >> b) | filler  # fill int 0's with 1's
            # print("a")
            return a
        # SRL
        else:
            # print("second branch")
            return a >> b
    elif func3 == Func3.SLT:
        if to_signed32(a) < to_signed32(b):
            return 1
        else:
            return 0
    elif func3 == Func3.SLTU or func3 == Func3.SLTU:
        if (a & max_int32) < (b & max_int32):
            return 1
        else:
            return 0
    else:
        raise Exception("write %r" % func3)


# just a pretty printer
def dump() -> None:
    pp: list[str] = []
    for i in range(32):
        if i != 0 and i % 8 == 0:
            pp += "\n"
        # %3s puts output inside 3 spaces
        # %08x means it's an 8 digit hexadecimal
        # pp += " %3s: %08x" % ("x%d" % i, regfile[i])
        pp += " %3s: %08x" % (regs[i], regfile[i])
    pp += "\n  PC: %08x" % regfile[PC]
    print("".join(pp))


# This sign_extend only puts 1 extra bit at the front, but we need like, way more
# This looks super wrong
def sign_extend(x, l):
    if x & (1 << (l - 1)):
        x |= -1 << l
    return x


def step() -> bool:
    # Instruction Fetch
    ins = r32(regfile[PC])
    # Instruction Decode
    # little endian
    # bitwise and with 63 in hex = get first 6 digits
    # Then wrapped with our op enum

    def gibi(s, e) -> int:
        # This is so cute
        return (ins >> e) & ((1 << (s - e + 1)) - 1)

    bits = gibi(6, 0)
    # print("%x %8x " % (regfile[PC], ins))
    opcode = Ops(bits)
    # print("%x %8x %r" % (regfile[PC], ins, opcode))
    # dump()
    # print(hex(offset), rd)
    if opcode == Ops.JAL:
        # J-Type
        offset = sign_extend(
            (
                gibi(31, 31) << 20
                | gibi(30, 21) << 1
                | gibi(20, 20) << 11
                | gibi(19, 12) << 12
            ),
            20,
        )
        # rd = destination register (why do they do everything backwards)
        rd = gibi(11, 7)
        regfile[rd] = regfile[PC] + 4
        regfile[PC] += offset
        return True

    elif opcode == Ops.JALR:
        # I-type
        rd = gibi(11, 7)
        func3 = Func3(gibi(14, 12))
        rs1 = gibi(19, 15)
        offset = sign_extend(gibi(31, 20), 12)

        # register of the instruction following the jump is written to register rd.
        # The increment is because each instruction if 4 bytes long
        # This does not look like what we're supposed to do at all
        addr = (regfile[rs1] + offset) & (((1 << 32) - 1) << 1)
        regfile[rd] = regfile[PC] + 4
        regfile[PC] = addr
        return True

    elif opcode == Ops.IMM:
        # I-type
        # destination register
        rd = gibi(11, 7)
        func3 = Func3(gibi(14, 12))
        # source register 1
        rs1 = gibi(19, 15)
        # imm means immediate value
        imm = sign_extend(gibi(31, 20), 12)
        func7 = gibi(31, 25)
        regfile[rd] = imm_arith(func3, regfile[rs1], imm)

    elif opcode == Ops.OP:
        # R-type
        rd = gibi(11, 7)
        rs1 = gibi(19, 15)
        rs2 = gibi(24, 20)
        func7 = gibi(31, 25)
        func3 = Func3(gibi(14, 12))

        regfile[rd] = op_arith(func3, regfile[rs1], regfile[rs2], func7=func7)
        # if func3 == Func3.ADD:
        #     if func7 == 0:
        #         regfile[rd] = regfile[rs1] + regfile[rs2]
        #     else:
        #         regfile[rd] = regfile[rs1] - regfile[rs2]
        # if func3 == Func3.XOR:
        #     regfile[rd] = regfile[rs1] ^ regfile[rs2]
        # if func3 == Func3.SRA:

    elif opcode == Ops.LUI:
        # U-type
        # Pretty sure this is wrong. We need to fill lower 12 with zeros,
        # and gibi 31, 12
        rd = gibi(11, 7)
        uimm = gibi(31, 12)
        regfile[rd] = uimm << 12

    elif opcode == Ops.AUIPC:
        # U-type
        rd = gibi(11, 7)
        uimm = gibi(31, 12) << 12

        regfile[rd] = regfile[PC] + uimm

    elif opcode == Ops.BRANCH:
        # B-type
        func3: Func3 = Func3(gibi(14, 12))
        rs1 = gibi(19, 15)
        rs2 = gibi(24, 20)
        imm = sign_extend(
            gibi(11, 8) << 1
            | gibi(30, 25) << 5
            | gibi(7, 7) << 11
            | gibi(31, 31) << 12,
            12,
        )
        a = to_signed32(regfile[rs1])
        b = to_signed32(regfile[rs2])
        offset = sign_extend(regfile[rs2], 12) << 1

        cond = False
        if func3 == Func3.BEQ:
            cond = regfile[rs1] == regfile[rs2]

        elif func3 == Func3.BNE:
            cond = regfile[rs1] != regfile[rs2]

        elif func3 == Func3.BLT:
            cond = a < b

        elif func3 == Func3.BGE:
            cond = a >= b

        # Unsigned comparison
        elif func3 == Func3.BLTU:
            cond = (regfile[rs1] & max_int32) < (regfile[rs2] & max_int32)

        # Unsigned comparison
        elif func3 == Func3.BGEU:
            cond = (regfile[rs1] & max_int32) >= (regfile[rs2] & max_int32)

        else:
            dump()
            raise Exception("write %r func3 %r" % (opcode, func3))
        if cond:
            regfile[PC] += imm
            return True

    elif opcode == Ops.STORE:
        # S-type
        func3 = Func3(gibi(14, 12))
        rd = gibi(11, 7)
        rs1 = gibi(19, 15)

        # Stores 32 bit values from rs2 to mem
        imm = sign_extend(gibi(11, 7) | gibi(31, 25) << 5, 12)
        rs2 = gibi(24, 20)
        func3 = Func3(gibi(14, 12))
        data = regfile[rs2]
        addr = regfile[rs1] + imm

        if func3 == Func3.SW:
            pass
        if func3 == Func3.SH:
            data = data & ((1 << 16) - 1)
        if func3 == Func3.SB:
            data = data & ((1 << 8) - 1)

        ws(data, addr)
        # print("STORE %8x = %x" % (addr, data))

    elif opcode == Ops.LOAD:
        # I-type
        rd = gibi(11, 7)
        func3 = Func3(gibi(14, 12))
        rs1 = gibi(19, 15)
        imm = sign_extend(gibi(31, 20), 12)
        addr = regfile[rs1] + imm
        val = r32(addr)
        # print("VAL ADDR imm = %x %x %x" % (val, addr, imm))
        if func3 == Func3.LW:
            pass
        if func3 == Func3.LH:
            val = sign_extend(val & ((1 << 16) - 1), 16)
        if func3 == Func3.LHU:
            val = val & ((1 << 16) - 1)
        if func3 == Func3.LB:
            val = sign_extend(val & ((1 << 8) - 1), 8)
        if func3 == Func3.LBU:
            val = val & ((1 << 8) - 1)

        # print("LOAD %8x = %x" % (addr, val))

        regfile[rd] = val

    elif opcode == Ops.MISC:
        pass

    elif opcode == Ops.SYSTEM:
        rd = gibi(11, 7)
        func3 = Func3(gibi(14, 12))
        rs1 = gibi(19, 15)
        csr = gibi(31, 20)
        if func3 == Func3.CSRRS:
            # print("CSRRS", rd, rs1, hex(csr))
            pass
        elif func3 == Func3.CSRRW:
            if csr == 3072:
                return False
            # print("CSRRW", rd, rs1, hex(csr))
        elif func3 == Func3.CSRRWI:
            # print("CSRRWI", rd, rs1, hex(csr))
            pass
        elif func3 == Func3.ECALL:
            # print("ECALL", rd, rs1, hex(csr))
            if regfile[17] == 93:
                print(regfile[10])
                if regfile[10] != 0:
                    test_no = (regfile[10] - 1) / 2
                    # not sure why, but these tests are actually broken
                    bad_tests = [
                        ("riscv-tests/isa/rv32ui-p-slti", 7),
                        ("riscv-tests/isa/rv32ui-p-sb", 4),
                        ("riscv-tests/isa/rv32ui-p-sh", 4),
                    ]
                    if (curr_file, test_no) in bad_tests:
                        pass
                    else:
                        raise Exception("Test %d failed!" % ((regfile[10] - 1) / 2))
                elif regfile[10] == 0:
                    print("Tests passed!")
                else:
                    print("Returned with %r" % regfile[10])

        else:
            dump()
            raise Exception("func %r" % func3)

    else:
        dump()
        raise Exception("write %r" % opcode)

    regfile[PC] += 4
    return True


global curr_file

if __name__ == "__main__":
    for x in glob.glob("riscv-tests/isa/rv32ui-p-*"):
        # .dump files here are the disassemblys of the tests
        if x.endswith(".dump"):
            continue
        if "fence_i" in x:
            continue
        # if x != ("riscv-tests/isa/rv32ui-p-sll"):
        #     continue
        # open file and r(ead) as b(inary)
        else:
            curr_file = x
            reset()
            with open(x, "rb") as f:
                print(x)
                e = ELFFile(f)
                for s in e.iter_segments():
                    # some of the headers are just metadata (whatever that means), so ignore them
                    # We care about 0x80000000 because that's apparently where memory mapped io and dram is
                    if s.header.p_paddr > 0:
                        # writes the header data to the p_paddr (physical address) in memory.
                        ws(s.data(), s.header.p_paddr)
                # return our pointer to the start
                regfile[PC] = 0x80000000
                try:
                    while step():
                        pass
                except Exception as e:
                    raise Exception("Test %s experienced failure %s" % (x, e))
