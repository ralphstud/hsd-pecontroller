from amaranth import *
from pe import PE
from adder_tree import AdderTree
from enum import IntEnum


def is_power_of_two(x):
    return (x & (x - 1)) == 0


class ACTCODE(IntEnum):
    NONE = 0
    RELU = 1


class PEStack(Elaboratable):
    def __init__(self, num_bits, width, cnt_bits, signed=True):
        self.width = width  # input bitwidth
        self.acc_bits = width
        self.num_stack = width // num_bits
        self.num_bits = num_bits
        self.cnt_bits = cnt_bits
        self.signed = signed

        assert width in [32, 64, 128]
        assert width % num_bits == 0
        assert is_power_of_two(self.num_stack)

        self.adder_tree = AdderTree(
            acc_bits=self.acc_bits, fan_in=self.num_stack, signed=signed)

        self.pe_arr = [
            PE(num_bits=num_bits, acc_bits=self.acc_bits,
               cnt_bits=cnt_bits, signed=signed)
            for _ in range(self.num_stack)]

        self.in_rst = Signal(1, reset_less=True)
        self.in_init = Signal(cnt_bits)
        self.in_a = Signal(width)
        self.in_b = Signal(width)
        self.in_act = Signal(1)

        self.out_d = Signal(Shape(self.acc_bits, signed=True))
        self.out_ready = Signal(1)
        self.out_ovf = Signal(1)

    def elaborate(self, platform):
        m = Module()

        m.submodules.adder_tree = adder_tree = self.adder_tree

        # TODO
        for i in range(self.num_stack):
            m.d.comb += [
                adder_tree.in_data[i].eq(self.pe_arr[i].out_d),
                adder_tree.in_valid[i].eq(self.pe_arr[i].out_d_valid),
                adder_tree.in_ovf[i].eq(self.pe_arr[i].out_ovf)
            ]
        m.d.comb += [
            self.out_d.eq(adder_tree.out_d),
            self.out_ready.eq(adder_tree.out_valid),
            self.out_ovf.eq(adder_tree.out_ovf)
        ]

        for i, pe in enumerate(self.pe_arr):
            m.submodules += pe

            # TODO
            m.d.comb += [
                pe.in_a.eq(self.in_a[self.num_bits * i:self.num_bits + self.num_bits * i - 1]),
                pe.in_b.eq(self.in_b[self.num_bits * i:self.num_bits + self.num_bits * i - 1]),
                pe.in_init.eq(self.in_init),
                pe.in_rst.eq(self.in_rst)
            ]

        return m


if __name__ == '__main__':
    num_bits = 8
    width = 32
    cnt_bits = 5
    signed = True
    dut = PEStack(num_bits=num_bits, width=width,
                  cnt_bits=cnt_bits, signed=signed)
    dut = ResetInserter(dut.in_rst)(dut)

    from amaranth.sim import Simulator
    import numpy as np

    np.random.seed(42)

    def test_case(dut, in_a, in_b, in_init):
        yield dut.in_a.eq(in_a)
        yield dut.in_b.eq(in_b)
        yield dut.in_init.eq(in_init)
        yield

    def gen_vec(len_vec):
        arr = []
        low = 0
        high = 2 ** num_bits
        for _ in range(len_vec):
            tmp = 0
            for _ in range(width // num_bits):
                tmp = (tmp << num_bits) +\
                    int(np.random.randint(low=low, high=high))
            arr.append(tmp)
        return arr

    def bench():
        num_try = 4
        for i in range(num_try):
            i_stack = []
            j_stack = []
            len_vec = np.random.randint(low=1, high=2**cnt_bits)
            i_stack = gen_vec(len_vec)
            j_stack = gen_vec(len_vec)

            # initialize
            yield from test_case(dut, 0, 0, len_vec)
            # feed
            for i in range(len_vec):
                yield from test_case(dut, i_stack[i], j_stack[i], 0)

    from pathlib import Path
    p = Path(__file__)

    sim = Simulator(dut)
    sim.add_clock(1e-6)
    sim.add_sync_process(bench)

    with open(p.with_suffix('.vcd'), 'w') as f:
        with sim.write_vcd(f):
            sim.run()

    from amaranth.back import verilog
    top = PEStack(num_bits=num_bits, width=width,
                  cnt_bits=cnt_bits, signed=signed)
    with open(p.with_suffix('.v'), 'w') as f:
        f.write(
            verilog.convert(
                top,
                ports=[
                    top.in_a, top.in_b, top.in_init,
                    top.out_d, top.out_ready, top.out_ovf]))