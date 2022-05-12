from math import ceil

def get_symbol(i):
    return 'v%d' % (i);
def gen_args(l):
    args = '(' + ','.join([get_symbol(i) for i in l]) + ')';
    return args

for i in range(7,173):
    half = int(i / 2)
    ceiled = ceil(i/2)
    args = gen_args(range(i));
    define = "#define minmax%d%s " % (i, args);
    define += "minmax%d%s;minmax%d%s;mm2(%s,%s);mm2(%s,%s);" % (half,gen_args(range(half)),ceiled,gen_args(range(half,i)),get_symbol(0),get_symbol(half),get_symbol(half-1),get_symbol(i-1));
    print(define)

def gen_minmax(i):
    if i > 127:
        half = int(i / 2)
        ceiled = ceil(i/2)
        call =  "minmax%d(%s);\n" % (half, ','.join(["array+%d" % (j) for j in range(half)]))
        call +=  "minmax%d(%s);\n" % (ceiled, ','.join(["array+%d" % (j) for j in range(half,i)]))
        call += "mm2(array+%d,array+%d);mm2(array+%d,array+%d);\n" % (0,half, half-1,i-1)
        return call
    else:
        args = ','.join(["array+%d" % (j) for j in range(0,i)])
        return "minmax%d(%s);\n" % (i,args);
def gen_switch(n):
    for i in range(3,n):
        case = "case %d:\n" % (i);
        case += gen_minmax(i);
        case += "return;"
        print(case)

print()
print()
print()
gen_switch(64)
gen_switch(174)
