import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

x=np.arange(-10,10,0.2)#deivied the no from -10to 10 by the difference of 0.2
sig=1/(1+np.power(np.e,-x))#creat the sigmode function
line=4*x+7
sig_1=(np.power(np.e,-x))/(1+np.power(np.e,-x))#creat the another sig mode function 
plt.plot(x,sig)
plt.show()

plt.plot(x,sig_1)
plt.show()

plt.plot(x,line)
plt.show()

sig_line=1/(1+np.power(np.e,-line))#line pass through sig functon
plt.plot(x,sig_line)
plt.show()
 