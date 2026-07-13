import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--SERIAL_PORT1", type=str, default="COM5", help="第一个报警器的串口号")
    parser.add_argument("--area_thred", type=int, default=1500, help="物体面积阈值")
    parser.add_argument("--confid_level", type=float, default=0.8, help="识别置信度")
    parser.add_argument("--aaa", type=int, default=100)
    parser.add_argument("-b", "--bbb", type=int, default=10)
    opt = parser.parse_args()
    a = opt.aaa
    b = opt.bbb
    print(a + b)