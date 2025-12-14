#!/usr/bin/env python3

import nmrglue as ng
import matplotlib.pyplot as plt

#xl = [14.5, 10.3]
#yl = [0, 12000000]

plt.figure(figsize=(11, 8.5))  # landscape

dic, data = ng.sparky.read('./sim.ucsf')
uc = ng.sparky.make_uc(dic, data, dim=1)
plt.plot(uc.ppm_scale(), data.max(0), 'k-')
plt.gca().invert_xaxis()
plt.xlabel('1H (ppm)')
plt.ylabel('Intensity')
plt.title('1H Projection of Imino Spectrum')
#plt.xlim(xl)
#plt.ylim(yl)

plt.savefig('proj.pdf')
plt.show()
