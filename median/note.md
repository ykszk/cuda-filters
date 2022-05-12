## 7x7x7 ##
filter size : $s_f = 7$  
total size : $s = s_f^3$ = 343  
required buffer size : $s_r = \lceil\frac{s}{2}\rceil+1 = 173$

### 2 pix ###
overlap : $o = 7*7*6=294$   
common reduction : $c = o - s_r = 121$  
temporary buffer size : $s_r - c -1 = 51$

### 3 pix ###
overlap : $o = 7*7*5=245$   
common reduction : $c = o - s_r = 72$  
temporary buffer size : $s_r - c -1 = 100$