import sys

from TensorflowToolbox.utility import read_proto as rp
from TensorflowToolbox.utility import file_io

import params
import net_flow


if __name__ == "__main__":
    net = net_flow.NetFlow(params, True, True)
    net.mainloop()
