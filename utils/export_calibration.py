import data.calibration_constants as c

def printMatrix(name, matrix):
    print "%s: !!opencv-matrix" % name 
    print "   rows: %d" % matrix.shape[0]
    print "   cols: %d" % matrix.shape[1]
    print "   dt: d"
    print "   data:", repr(matrix.flatten().tolist())

print "%YAML:1.0"

printMatrix("Ml", c.Ml)
printMatrix("dl", c.dl)
printMatrix("Mr", c.Mr)
printMatrix("dr", c.dr)
printMatrix("R", c.R)
printMatrix("T", c.T)
