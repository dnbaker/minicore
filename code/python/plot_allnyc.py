import matplotlib
import matplotlib.pyplot as plt
import numpy as np

EXAMPLE_DATA = \
'''
-73.9984	40.7287
-74.0071	40.7099
-73.9764	40.738
-74.048	40.7021
-73.948	40.7244
-73.9511	40.737
-74.0153	40.7185
-73.9781	40.7117
-73.9879	40.7174
-74.0052	40.7121
-73.9356	40.7369
-73.998	40.7302
-74.0009	40.7117
-74.0047	40.7347
-73.9476	40.7232
-73.9876	40.7038
-73.979	40.7123
-73.925	40.7325
-73.9984	40.7308
-73.9295	40.7331
-74.007	40.7218
-74.0014	40.6759
-73.9518	40.7155
-73.9245	40.7356
-74.0092	40.7023
-74.052	40.7313
-73.9389	40.7338
-73.9602	40.71
-73.9785	40.7115
-73.9922	40.7379
-74.0053	40.706
-73.9996	40.731
-73.9798	40.735
-73.9949	40.7231
-74.0104	40.7305
-73.9908	40.7354
-74.0056	40.712
-73.9924	40.7122
-73.9229	40.7085
-73.9927	40.7025
'''

def parse_data(lines):
    lons, lats = [], []
    for i in (j.split() for j in lines.strip().split('\n')):
        lon = float(i[0])
        lat = float(i[1])
        lons.append(lon)
        lats.append(lat)
    return tuple(map(np.array, (lats, lons)))

if __name__ == '__main__':
    import sys
    
    fn = sys.argv[1]
    lons, lats = parse_data(open(sys.argv[1]).read())
    color = ['b'] * len(lons)
    marker = '.'
    bbox = None
    inpts, outpts = [], []
    if sys.argv[2:]:
        #-74.027596,40.701724,-73.920479,40.880295
        bbox = list(map(float, sys.argv[2].split(",")))[:4]
        if len(bbox) != 4: raise Exception("the Roof")
        for ind in range(len(lons)):
            if lons[ind] > bbox[0] and lons[ind] < bbox[2] and \
                lats[ind] > bbox[1] and lats[ind] < bbox[3]:
                    inpts.append(ind)
                    color[ind] = 'r'
            else:
                outpts.append(ind)
    else:
        outpts = list(range(len(lons)))
    #plt.scatter(x=lons[inpts], y=lats[inpts], alpha=.9, s=.1, c='r', marker='.')
    #plt.scatter(x=lons[outpts], y=lats[outpts], alpha=.9, s=1.5, c='k', marker='o')
    print(f"lons: {lons[:5]}")
    print(f"lats: {lats[:5]}")
    plt.scatter(y=lats, x=lons, alpha=0.6, s=.01, c='k', marker='.')
    plt.ylabel("Latitude")
    plt.xlabel("Longitude")
    #plt.title("New York City")
    ofname = f"{fn.split('/')[-1].split('.')[0]}.png"
    plt.savefig(ofname)
    plt.savefig(ofname.replace("png", "svg"))
