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
    return tuple(map(np.array, (lons, lats)))

if __name__ == '__main__':
    import sys
    import argparse
    ap = argparse.ArgumentParser()
    aa = ap.add_argument
    aa("coords")
    aa("-B", "--bounding-box", default=None)
    aa("--title")
    aa("--font-size", "-f", default=16, type=int)
    args = ap.parse_args()
    fontsize = args.font_size
    fn = args.coords
    lons, lats = parse_data(open(fn).read())
    
    #title = sys.argv[2] if sys.argv[2:] else "Dataset: " + sys.argv[1].split("/")[-1].split(".")[0]
    color = ['k'] * len(lons)
    marker = '.'
    bbox = None
    inpts, outpts = [], []
    if args.bounding_box:
        bbox = list(map(float, args.bounding_box.split(",")))[:4]
        if len(bbox) != 4: raise Exception("the Roof")
        for ind in range(len(lons)):
            if lons[ind] > bbox[0] and lons[ind] < bbox[2] and \
                lats[ind] > bbox[1] and lats[ind] < bbox[3]:
                    inpts.append(ind)
                    color[ind] = 'b'
            else:
                outpts.append(ind)
        rep = {'b': 'k', 'k': 'b'}
        color = [rep[c] for c in color]
    else:
        outpts = list(range(len(lons)))
    #plt.scatter(x=lons[inpts], y=lats[inpts], alpha=.9, s=.1, c='r', marker='.')
    #plt.scatter(x=lons[outpts], y=lats[outpts], alpha=.9, s=1.5, c='k', marker='o')
    print("lats: %s" % lats[:5])
    print("lons: %s" % lons[:5])
    #lons, lats = lats, lons
    plt.scatter(x=lons, y=lats, alpha=0.9, s=.75, c=color, marker='x')
    plt.xlabel("Longitude", fontsize=fontsize)
    plt.ylabel("Latitude", fontsize=fontsize)
    #title = args.title if args.title else "Dataset: %s" % args.coords.split("/")[-1].split(".")[0]
    #plt.title(title)
    ofname = f"{fn.split('/')[-1].split('.')[0]}.{fontsize}.svg"
    plt.savefig(ofname)
    plt.savefig(ofname.replace("svg", "png"))
        
