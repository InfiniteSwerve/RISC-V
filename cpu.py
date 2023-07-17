#!/opt/miniconda3/bin/python3
"""
tests from 
https://github.com/riscv-software-src/riscv-tests
"""
# Interpret bytes as packed binary data
# Converts between python values and c structs represented as bytes
import struct

# Finds pathnames matching a pattern. stuff like ../../Downloads/*/*.exe
import glob

# elf = Executable and Linkable Format = Super versatile form of an executable file used
from elftools.elf.elffile import ELFFile
from enum import Enum


# Our set of registers.
class Regfile:
    def __init__(self):
        self.regs = [0] * 33

    def __getitem__(self, x):
        return self.regs[x]

    def __setitem__(self, key, value):
        # MIPS convention to have register 0 always be zero (that's why we have 33 registers)
        if key == 0:
            return
        # Mask value to make sure we only grab 32 bits of it
        self.regs[key] = value & 0xFFFFFFFF


regfile = Regfile()
# register file, as in, keeps track of all register values
PC = 32


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

    # SYSTEM
    # I-type
    # Doesn't do anything for us, but,
    # ECALL makes a 'service request to the system environment'
    # EBREAK returns control
    ECALL = EBREAK = 0b000


# 64k at 0x80000000
memory = b"\x00" * 0x10000


def ws(dat, addr):
    global memory
    addr -= 0x80000000
    assert addr >= 0 and addr < len(memory)
    # won't this increase total memory size over time?
    memory = memory[:addr] + dat + memory[addr + len(dat) :]


# Reads 32 bits from address
# I = interpret as unsigned int
# Unpack always returns a tuple even if there's no other elements
def r32(addr):
    addr -= 0x80000000
    if addr < 0 or addr > len(memory):
        raise Exception("read out of bounds: %x" % addr)
    return struct.unpack("<I", memory[addr : addr + 4])[0]


# just a pretty printer
def dump() -> None:
    pp: list[str] = []
    for i in range(32):
        if i != 0 and i % 8 == 0:
            pp += "\n"
        # %3s puts output inside 3 spaces
        # %08x means it's an 8 digit hexadecimal
        pp += " %3s: %08x" % ("x%d" % i, regfile[i])
    pp += "\n  PC: %08x" % regfile[PC]
    print("".join(pp))


def sign_extend(x, l):
    if x >> (l - 1) == 1:
        return (1 << l) - x
    else:
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
    print("%x %8x %r" % (regfile[PC], ins, opcode))
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
        func3 = gibi(14, 12)
        rs1 = gibi(19, 15)
        offset = sign_extend(gibi(31, 20), 12)
        # register of the instruction following the jump is written to register rd.
        # The increment is because each instruction if 4 bytes long
        regfile[rd] += 4
        # This does not look like what we're supposed to do at all
        regfile[PC] = regfile[rs1] + offset
        return True

    elif opcode == Ops.IMM:
        # I-type
        # destination register
        rd = gibi(11, 7)
        func3 = Func3(gibi(14, 12))
        # source register 1
        rs1 = gibi(19, 15)
        # imm means immediate value
        imm = gibi(31, 20)
        # print(rd, rs1, func3, imm)
        if func3 == Func3.ADDI:
            regfile[rd] = regfile[rs1] + imm
        elif func3 == Func3.SLLI:
            regfile[rd] = regfile[rs1] << imm
        elif func3 == Func3.SRLI:
            regfile[rd] = regfile[rs1] >> imm

        else:
            raise Exception("write %r" % func3)

    elif opcode == Ops.OP:
        # R-type
        rd = gibi(11, 7)
        rs1 = gibi(19, 15)
        rs2 = gibi(24, 20)
        func7 = gibi(31, 25)
        func3 = Func3(gibi(14, 12))

        if func3 == Func3.ADD:
            regfile[rd] = regfile[rs1] + regfile[rs2]

    elif opcode == Ops.AUIPC:
        # J-type
        rd = gibi(11, 7)
        uimm = gibi(31, 20)
        regfile[rd] = regfile[PC] + uimm

    elif opcode == Ops.BRANCH:
        # B-type
        func3 = Func3(gibi(14, 12))
        rs1 = gibi(19, 15)
        rs2 = gibi(24, 20)
        imm = sign_extend(
            gibi(11, 8) << 1
            | gibi(30, 25) << 6
            | gibi(7, 7) << 11
            | gibi(31, 31) << 12,
            12,
        )

        print(func3)
        if func3 == Func3.BNE:
            if regfile[rs1] != sign_extend(regfile[rs2], 12) << 1:
                regfile[PC] += imm
            else:
                regfile[PC] += 4

    elif opcode == Ops.SYSTEM:
        pass

    else:
        dump()
        raise Exception("write %r" % opcode)

    regfile[PC] += 4
    return True

    dump()
    # Execute
    # Access
    # Write-Back
    return False


if __name__ == "__main__":
    for x in glob.glob("riscv-tests/isa/rv32ui-*"):
        # .dump files here are the disassemblys of the tests
        if x.endswith(".dump"):
            continue
        # open file and r(ead) as b(inary)
        if x != "riscv-tests/isa/rv32ui-v-add":
            continue
        else:
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
                while step():
                    pass
            break
