import numpy as np

################################################################################
class BCTab:
    """Chemical shift lookup table parser for RNA motif-based prediction."""

    def __init__(self, tab=None):
        self.CH = {}
        self.NH = {}
        self.CHnu = {}
        self.NHnu = {}
        self.CHHz = {}
        self.NHHz = {}
        self.N = {}
        self.H = {}
        self.C = {}
        self.num = {}
        if tab:
            self.readTab(tab)

    def readTab(self, tab):
        lines = [line.strip() for line in open(tab).readlines()[1:]]
        Tab1 = {x.split()[0]: float(x.split()[1]) for x in lines}
        Tab2 = {x.split()[0]: float(x.split()[2]) for x in lines}
        try:
            self.num = {x.split()[0]: int(x.split()[3]) for x in lines}
        except IndexError:
            pass
        bds = sorted(Tab1.keys())
        tab2_vals = np.array(list(Tab2.values()))
        tab1_vals = np.array(list(Tab1.values()))

        if np.all(tab2_vals > 9):
            self.N = {x: Tab1[x] for x in bds}
            self.H = {x: Tab2[x] for x in bds}
            self.NH = {x: [Tab1[x], Tab2[x]] for x in bds}
            self.NHHz = {x: [Tab1[x]*60, Tab2[x]*600] for x in bds}
            for x in bds:
                if x[0] in ['G', 'A']:
                    self.NHnu[x] = {'N1': Tab1[x], 'H1': Tab2[x]}
                if x[0] in ['U', 'C']:
                    self.NHnu[x] = {'N3': Tab1[x], 'H3': Tab2[x]}
        elif (np.all(tab2_vals < 8.6) and np.all(6.4 < tab2_vals) and
              np.all(tab1_vals < 148) and np.all(134 < tab1_vals)):
            self.C = {x: Tab1[x] for x in bds}
            self.H = {x: Tab2[x] for x in bds}
            self.CH = {x: [Tab1[x], Tab2[x]] for x in bds}
            self.CHHz = {x: [Tab1[x]*150, Tab2[x]*600] for x in bds}
            for x in bds:
                if x[0] in ['G', 'A']:
                    self.CHnu[x] = {'C8': Tab1[x], 'H8': Tab2[x]}
                if x[0] in ['U', 'C']:
                    self.CHnu[x] = {'C6': Tab1[x], 'H6': Tab2[x]}

        elif (np.all(tab2_vals < 6.4) and np.all(4.8 < tab2_vals) and
              np.all(tab1_vals < 98) and np.all(94 < tab1_vals)):
            self.C = {x: Tab1[x] for x in bds}
            self.H = {x: Tab2[x] for x in bds}
            self.CH = {x: [Tab1[x], Tab2[x]] for x in bds}
            self.CHHz = {x: [Tab1[x]*150, Tab2[x]*600] for x in bds}
            for x in bds:
                self.CHnu[x] = {"C1'": Tab1[x], "H1'": Tab2[x]}
        elif (np.all(tab2_vals < 8.4) and np.all(6.0 < tab2_vals) and
              np.all(148 < tab1_vals) and np.all(tab1_vals < 158)):
            self.C = {x: Tab1[x] for x in bds}
            self.H = {x: Tab2[x] for x in bds}
            self.CH = {x: [Tab1[x], Tab2[x]] for x in bds}
            self.CHHz = {x: [Tab1[x]*150, Tab2[x]*600] for x in bds}
            for x in bds:
                self.CHnu[x] = {"C2": Tab1[x], "H2": Tab2[x]}


      
        

            

