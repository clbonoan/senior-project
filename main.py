import sys
import cannyedge
import hough
import lbp

if len(sys.argv) > 2:
    arg1 = sys.argv[1]
    arg2 = sys.argv[2]

    match(arg1):
        case "canny":
            cannyedge.canny(arg2)

        case "hough":
            hough.hough(arg2)    

        case "lbp":
            lbp.lbp(arg2)
            
        case _:
            print(f"Unknown operation: {arg1}")

else:
    print(f"arg1 [algorithm name] arg2 [image file name (jpg or jpeg ONLY)]")