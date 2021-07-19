import os
import sys
import atexit

thirdparty = os.path.dirname(__file__)
sys.path.append(os.path.join(thirdparty, "matrixlstm/classification"))
sys.path.append(os.path.join(thirdparty, "e2vid"))

from evrepr.thirdparty.e2vid.utils import timers
atexit.unregister(timers.print_timing_info)
