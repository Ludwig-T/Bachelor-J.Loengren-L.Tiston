import cdflib
import numpy as np
import matplotlib.pyplot as plt


FILE_PATH = 'C:/Users/joarl/OneDrive/Dokument/Skola/Kand/Data/L1_2020_06/11/solo_L1_rpw-lfr-surv-cwf-cdag_20200611_V10.cdf'


c = cdflib.CDF(FILE_PATH)



#print(c.cdf_info()) #general info
#print(c.varattsget('V')) #info about specific z-variable 
#c.print_attrs() #to get information of attributes

### Electrical field plot
E = c['E'] #NO UNITS
SR = c['SAMPLING_RATE'] #units in HZ (constant)
E1 = E[:, 0]
E2 = E[:, 1]
dt = 1/SR[0]
T_stop = len(E) * dt
X = np.arange(0, T_stop, dt)

plt.figure(1)
plt.plot(X, E1, label = 'E1')
plt.plot(X, E2, label = 'E2')
plt.xlabel('Time(S)')
plt.ylabel('Electrical fiel, NO UNITS')
plt.title('Plot of electrical field LF-data')
plt.legend()
plt.axhline(y=0, color = "black", linestyle='-')
plt.show()


### V-plot

V = c['V'] #NO UNITS
SR = c['SAMPLING_RATE'] #units in HZ (constant)
dt_V = 1/SR[0]
T_stop_V = len(V) * dt_V
X_V = np.arange(0, T_stop_V, dt_V)

plt.figure(2)
plt.plot(X_V, V, label = 'V')
plt.xlabel('Time(S)')
plt.ylabel('Potential V, NO UNITS')
plt.title('Plot of potential in LF-data')
plt.legend()
plt.axhline(y=0, color = "black", linestyle='-')
plt.show()
