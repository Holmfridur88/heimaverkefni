# mynd hugmynd 5

phi1_IV = lambda x: beta1*(np.sin(np.degrees(18*(np.pi**2)/a[1]*(x)))*5**4 + 50*np.sin(x**6)+ 58*x**5+ 4**5*np.sinh(x)*np.sin(x))     # bottom
phi2_IV = lambda x: -(np.cos(np.degrees(15*np.pi/8*(x)))*5**4 - 58*x**7+ 4**5*np.sinh(x)*np.sin(x))
X, Y, Z = varmajafnvaegi(h[1], a[1], b, beta1, beta2[1], phi1_IV, phi2_IV)
#plt.figure(3)
plot_surf(X, Y, Z)