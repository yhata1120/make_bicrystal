from cellcalc import get_primitive_hkl, MID, get_pri_vec_inplane
from interface_generator import core, print_near_axis, convert_vector_index
from csl_generator import print_list, getsigmas, get_theta_m_n_list
from numpy import array
from numpy import array, dot, round
from numpy.linalg import inv, norm
import numpy as np
import glob
import shutil
import os
from cellcalc import get_primitive_hkl, get_pri_vec_inplane, get_normal_index, MID
from interface_generator import core, print_near_axis, convert_vector_index, write_trans_file
from numpy import array, dot, round, cross, ceil
from numpy.linalg import inv, det, norm
from numpy import cross, dot, ceil
from numpy.linalg import norm, inv

import numpy as np
import pandas as pd
import shutil
import os
import math
import glob
from scipy.spatial import distance
import matplotlib.pyplot as plt
from numpy.linalg import inv, norm
from numpy import dot

from Element import Element

import sys
from abc import ABCMeta, abstractclassmethod

def read_atominfile(dirname):
    #read atom positions from POSCAR
    with open (f"{dirname}/atominfile",'r') as f:
        lines = f.readlines()
    atoms = np.array([0,0,0,0,0])
    for i in range(13,len(lines)):
        atoms = np.vstack((atoms,np.array(lines[i].split(),dtype = float)))
    atoms = np.delete(atoms,0,axis= 0)
    atoms = np.delete(atoms, [1], 1)
    a = float(lines[5].split()[1])
    b = float(lines[6].split()[1])
    c = float(lines[7].split()[1])
    return atoms, a, b, c

def read_atomin_id(dirname):
    #read atom positions from POSCAR
    with open (f"{dirname}/atominfile",'r') as f:
        lines = f.readlines()
    atoms = np.array([0,0,0,0,0])
    for i in range(13,len(lines)):
        atoms = np.vstack((atoms,np.array(lines[i].split(),dtype = float)))
    atoms = np.delete(atoms,0,axis= 0)
    a = float(lines[5].split()[1])
    b = float(lines[6].split()[1])
    c = float(lines[7].split()[1])
    return atoms

def getNearestValue(atoms, coodinates):
    """
    概要: リストからある値に最も近い値を返却する関数
    @param coodinates: 表面原子の座標
    @param atoms: セルの原子データ配列
    @param indice: 表面原子のインデックス
    """
    indice = []
    atoms_coodinates = atoms[:,2:5]
    for i in coodinates:
        # coodinates要素と対象値の差分を計算し最小値のインデックスを取得
        idx = norm(atoms_coodinates - i, axis = 1).argmin()
        indice.append(idx)
    return indice

class Surface:
    def __init__(self,LatP,bond_length,coordinates,cut_off):
        self.LatP = LatP
        self.bond_length = bond_length
        self.coordinates = coordinates
        self.cutoff_top = cut_off

        self.supercell = np.eye(3)
        self.supercell_atoms = np.eye(3)
        self.a = 0
        self.b = 0
        self.c = 0
        self.atoms_surface = np.eye(3)
        self.atoms_left = np.eye(3)
        self.atoms_right = np.eye(3)
        self.atoms_surface_m = np.eye(3)
        self.atoms_rbt = np.eye(3)
        self.supercell_atoms_r = np.eye(3)
        self.supercell_atoms_l = np.eye(3)
        self.supercell_atoms_c = np.eye(3)
        self.supercell_atoms_rbt = np.eye(3)
        self.supercell_atoms_id = np.eye(3)
        self.interface = np.eye(3)

        
    
    def generate(self,dirname):
        tol = 1e-5
        self.supercell_atoms, self.a, self.b, self.c = read_atominfile(dirname)
        self.supercell_atoms = self.supercell_atoms[:,1:4]
        supercell_atoms_copy = self.supercell_atoms.copy()
#         self.supercell_atoms = self.supercell_atoms[self.supercell_atoms[:,0]<self.a/2 + tol]
        self.supercell_atoms_r = supercell_atoms_copy[supercell_atoms_copy[:,0]>self.a/2 + tol]
        self.supercell_atoms_l = supercell_atoms_copy[supercell_atoms_copy[:,0]<self.a/2 - tol]
        self.supercell_atoms_c = supercell_atoms_copy[(self.a/2-tol<supercell_atoms_copy[:,0]) & (supercell_atoms_copy[:,0]<self.a/2+tol)]
        self.supercell_atoms_id = read_atomin_id2(dirname)
        
        

    def Write_to_lammps(self,supercell_atoms,filename):
        dim = np.array([1,1,1])
        X = supercell_atoms.copy()

        NumberAt = len(X) 

        dimx, dimy, dimz = dim

        xlo = 0.00000000
        xhi = self.a
        ylo = 0.00000000
        yhi = self.b 
        zlo = 0.00000000
        zhi = self.c

        yz = 0.0

        Type1 = np.ones(len(X), int).reshape(1, -1)

        Counter = np.arange(1, NumberAt + 1).reshape(1, -1)

        # data = np.concatenate((X_new, Y_new))
        W1 = np.concatenate((Type1.T, X), axis=1)
        FinalMat = np.concatenate((Counter.T, W1), axis=1)

        with open(filename, 'w') as f:
            f.write('#Header \n \n')
            f.write('{} atoms \n \n'.format(NumberAt))
            f.write('2 atom types \n \n')
            f.write('{0:.8f} {1:.8f} xlo xhi \n'.format(xlo, xhi))
            f.write('{0:.8f} {1:.8f} ylo yhi \n'.format(ylo, yhi))
            f.write('{0:.8f} {1:.8f} zlo zhi \n\n'.format(zlo, zhi))
            f.write('{0:.8f} {1:.8f} {2:.8f} xy xz yz \n\n'.format(0, 0, yz))            
            f.write('Atoms \n \n')
            np.savetxt(f, FinalMat, fmt='%i %i %.8f %.8f %.8f')
        f.close()
        
    def Write_to_lammps_id(self,supercell_atoms,filename):
        dim = np.array([1,1,1])
        X = supercell_atoms.copy()

        NumberAt = len(X) 

        dimx, dimy, dimz = dim

        xlo = 0.00000000
        xhi = self.a
        ylo = 0.00000000
        yhi = self.b 
        zlo = 0.00000000
        zhi = self.c

        yz = 0.0


        with open(filename, 'w') as f:
            f.write('#Header \n \n')
            f.write('{} atoms \n \n'.format(NumberAt))
            f.write('2 atom types \n \n')
            f.write('{0:.8f} {1:.8f} xlo xhi \n'.format(xlo, xhi))
            f.write('{0:.8f} {1:.8f} ylo yhi \n'.format(ylo, yhi))
            f.write('{0:.8f} {1:.8f} zlo zhi \n\n'.format(zlo, zhi))
            f.write('{0:.8f} {1:.8f} {2:.8f} xy xz yz \n\n'.format(0, 0, yz))            
            f.write('Atoms \n \n')
            np.savetxt(f, X, fmt='%i %i %.8f %.8f %.8f')
        f.close()
        
        
    def Expand_Super_cell(self,Atoms):
        # cell の大きさを入力
        basis = np.array([[self.a,0,0,],
                         [0,self.b,0,],
                         [0,0,self.c,]])
        """
        --- modified by Yaoshu ---
        expands the smallest CSL unitcell to the given dimensions.
        """

        # 拡大したい倍率
        x = np.arange(1) #等倍
        y = np.arange(-1,2,1) #-1～1で３倍
        z = np.arange(-1,2,1) #-1～1で３倍

        indice = (np.stack(np.meshgrid(x, y, z)).T).reshape(len(x) * len(y) * len(z), 3)

        #1.get atoms
        Atoms = Atoms.repeat(len(indice),axis=0) + np.tile(np.dot(indice,basis),(len(Atoms),1))
        return Atoms
        
    def Extract_middle_atoms(self,expand_atoms):
        ylo = -2*self.LatP
        yhi = self.b + 2*self.LatP
        zlo = -2*self.LatP
        zhi = self.c + 2*self.LatP
        extracted_atoms = expand_atoms[((ylo) < expand_atoms[:,1]) & (expand_atoms[:,1]<(yhi)) & ((zlo) < expand_atoms[:,2]) & (expand_atoms[:,2]<(zhi))]
        return extracted_atoms
    
    def Extract_atoms(self,supercell_atoms):
        atoms_max = self.a
        # atoms はRBTごとに入力
        # atoms_middleはcnaを行う際の注目原子、extracted_atomsは相手原子
        atoms_cut = supercell_atoms[(supercell_atoms[:,0] > (atoms_max/2 - self.LatP*3)) & (supercell_atoms[:,0]< (self.LatP*3 + atoms_max/2))]
        expand_atoms = self.Expand_Super_cell(atoms_cut)
        extracted_atoms = self.Extract_middle_atoms(expand_atoms)
        
        return extracted_atoms
    
        
    def cna(self,atoms):
        atoms_max = self.a
        extracted_atoms = self.Extract_atoms(atoms)
        atoms_middle = atoms[(atoms[:,0] > (atoms_max/2 - self.LatP*2)) & (atoms[:,0]< (self.LatP*2 + atoms_max/2))]
#        write_middle(atoms_middle,'middle_region')
        # 距離行列を計算
        dist_M = distance.cdist(atoms_middle, extracted_atoms, metric='euclidean')
        # 0を除去
#        print(dist_M.size)
#        print(len(dist_M[dist_M!=0]),len(atoms_middle))
        dist_M = dist_M[dist_M!=0].reshape(dist_M.shape[0],dist_M.shape[1]-1)
        # 距離行列をもとにダングリングボンドを持つ原子を抽出

        cnaed_atoms = atoms_middle[np.where(np.count_nonzero(dist_M < self.bond_length*self.cutoff_top, axis=1)<self.coordinates)]
        
        
        return cnaed_atoms
    
    def make_mirror(self,atoms_surface):
        tol = 1e-5

        supercell = np.array([[self.a/2,0,0,],
                             [0,self.b,0,],
                             [0,0,self.c,],])
        atoms_fractional = dot(inv(supercell),atoms_surface.T).T
        atoms_mirror = atoms_fractional.copy()
        atoms_mirror = atoms_mirror[atoms_mirror[:,0]<1-tol]
        atoms_mirror[:,0] = 2-atoms_mirror[:,0]
        atoms_fractional = np.vstack((atoms_fractional,atoms_mirror))
        atoms_surface_m = dot(supercell,atoms_fractional.T).T

        return atoms_surface_m
    
    def get_surface(self):
        self.atoms_surface = self.cna(self.supercell_atoms)
        self.atoms_surface_m = self.make_mirror(self.atoms_surface)
        self.atoms_surface_l = self.cna(self.supercell_atoms_l)
        self.atoms_surface_r = self.cna(self.supercell_atoms_r)
        self.atoms_interface = np.vstack((self.atoms_surface_l,self.supercell_atoms_c,self.atoms_surface_r))
        
        
    def get_RBT(self,dy = 0,dz = 0):
        tol = 1e-5
        atoms_surface_m = self.atoms_surface_m
        a = self.a
        b = self.b
        c = self.c
        
        self.atoms_surface_m = self.make_mirror(self.atoms_surface)
        

        atoms_left = atoms_surface_m[atoms_surface_m[:,0] < a/2 + tol]

        atoms_right = atoms_surface_m[atoms_surface_m[:,0] > a/2 + tol]

        atoms_right[:,1] = atoms_right[:,1] + dy
        atoms_right[:,2] = atoms_right[:,2] + dz
        atoms_rbt = np.vstack((atoms_left,atoms_right))
        self.atoms_rbt = atoms_rbt.copy()
        self.supercell_atoms_r[:,1] = self.supercell_atoms_r[:,1] + dy
        self.supercell_atoms_r[:,2] = self.supercell_atoms_r[:,2] + dz
        self.supercell_atoms_rbt = np.vstack((self.supercell_atoms_l,self.supercell_atoms_r))
        
        
        return atoms_rbt
    
    def show_picture_surface(self):
        atoms_rbt = self.atoms_rbt.copy()
        a = self.a
        b = self.b
        c = self.c
        atoms_left = atoms_rbt[atoms_rbt[:,0] < a/2 + 1e-5]
        atoms_right = atoms_rbt[atoms_rbt[:,0] > a/2 + 1e-5]

        y = atoms_left[:,1]
        z = atoms_left[:,2]
        x = -(a/2 - atoms_left[:,0])

        y_RBT = atoms_right[:,1]
        z_RBT = atoms_right[:,2]
        x_RBT = -(a/2 - atoms_right[:,0])

        fig, axes = plt.subplots(1, 2, tight_layout=True, gridspec_kw={ 'width_ratios': [6, 2]})
        #plt.figure(figsize = ((max(y) - min(y)),(max(z) - min(z))))

        axes[0] = axes[0]
        pcm = axes[0].scatter(y,z, c = abs(x), cmap = 'hot',s=100, vmin = 0, vmax = max(abs(x))+1)
        # fig.colorbar(pcm,cmap = 'hot')

        pcm = axes[0].scatter(y_RBT,z_RBT, c = abs(x_RBT), cmap = 'hot',s=50, vmin = 0,vmax = max(abs(x_RBT))+1)
        # fig.colorbar(pcm,cmap = 'hot')

        # plt.show()

        #plt.figure(figsize = (2*(max(x) - min(x)),(max(z) - min(z))))
        axes[1].grid()
        pcm = axes[1].scatter(x,z, c = abs(x), cmap = 'hot',s=100, vmin = 0, vmax = max(abs(x)) + 1)
        # fig.colorbar(pcm,cmap = 'hot')

        pcm = axes[1].scatter(x_RBT,z_RBT, c = abs(x_RBT), cmap = 'hot',s=50, vmin = 0, vmax = max(abs(x_RBT))+1)
        fig.colorbar(pcm)
        
        
    def get_atominfile2(self):
        """
        atoms: セル内の原子、id付きのもの、Surfaceの中ではself.supercell_atoms_id
        surface: class Surface なのでckassの中ではselfで代用
        """
        atoms = self.supercell_atoms_id.copy()
        
        # overwrite ids of atoms on interface
        atoms[getNearestValue(atoms,self.atoms_interface),1] = 2
        # get atominfile2, which is the atominfile whose interface atoms' ids have been overwritten
        self.Write_to_lammps_id(atoms,"atominfile2")

# def getLatP():
#         """
#         read the Lattice Parameter
#         """
#         with open('atomsout','r') as f:
#             lines=f.readlines()
#             LatP = lines[-5:][0].split()[4].replace(';', '')
#         return float(LatP)

def get_a_b(CSL, axis):
    hkl_perp_axis = MID(CSL, axis)
    a, b = get_pri_vec_inplane(hkl_perp_axis, CSL).T

    if(norm(cross(axis,[1,0,0])) < 1e-8):
        b = a + b
    elif (norm(cross(axis,[1,1,1])) < 1e-8):
        if dot(a,b) < 0:
            b = a + b
        b = a + b
    if (abs(norm(a) - norm(b)) < 1e-8):
        raise RuntimeError ('the tow vectors are identical!')

    return a.T, b.T

def get_STGB_MLs(CSL, n_1, n_2):
    hkl_1 = MID(CSL, n_1)
    hkl_2 = MID(CSL, n_2)

    return hkl_1, hkl_2

def get_expansion_xyz(cell):
    exp_x = ceil(100/norm(cell[:,0]))
    exp_y = ceil(20/norm(cell[:,1]))
    exp_z = ceil(20/norm(cell[:,2]))
    return exp_x, exp_y, exp_z

def overwrite_supercell(interface,axis_input,n):
    """
    axis_input: 自分が入れる回転軸のpremitive cellでの表示
    n: 粒界面を指定するベクトルの直交座標表示
    """
    axis_cart = dot(interface.lattice_1, axis_input)
    axis_int = np.array(np.round(dot(inv(interface.conv_lattice_1), axis_cart),8),dtype = int)

    n_1_int = np.array(np.round(dot(inv(interface.conv_lattice_1), n),8),dtype = int)
    n_1_pre = np.array(np.round(dot(inv(interface.lattice_1),n),8),dtype = int)
    v3 = cross(n,axis_int)
    #print(interface.lattice_1)
    v3_int = np.array(np.round(dot(inv(interface.lattice_1), v3),8),dtype = int)
    v3_int_reduced = v3_int/abs(np.gcd.reduce(v3_int))
    bicrystal_U1 = np.column_stack([n_1_pre,axis_input])
    bicrystal_U1 = np.column_stack([bicrystal_U1,v3_int_reduced])
    interface.bicrystal_U1 = bicrystal_U1
    supercell = dot(interface.lattice_1,bicrystal_U1)

    bicrystal_U2 = np.array(np.round(dot(inv(interface.lattice_2_TD), supercell),8),dtype = int)
    interface.bicrystal_U2 = bicrystal_U2

def get_gb_files(interface, hkl, axis, sigma, axis_name, hkl_name, ab, file, axis_num, ab_num, bond_length,n,surface,name,element,stgb):
    print(stgb)
    interface.compute_bicrystal(hkl, normal_ortho = True, plane_ortho = True, lim = 50, tol = 1e-10)
#    print(f'U1 is {interface.bicrystal_U1}')
    half_cell = dot(interface.lattice_1, interface.bicrystal_U1)
#    print(f'lattice1_2 is {interface.lattice_1}')
    x,y,z = get_expansion_xyz(half_cell)
    axis_x, axis_y, axis_z = axis_name
    axis_name_num = 100*axis_x + 10*axis_y +axis_z
    hkl_name = np.array(hkl_name, dtype=int)
    hkl_name_num = hkl_name.copy()
    hkl_name = np.array(np.sort(np.abs(hkl_name))[::-1]/np.gcd.reduce(np.sort(np.abs(hkl_name))[::-1]),dtype=int)
    hkl_x, hkl_y, hkl_z = hkl_name
#    dirname = 100*hkl_x + 10*hkl_y + hkl_z
    dirname = f"{str(hkl_x)}_{str(hkl_y)}_{str(hkl_z)}_{hkl_name_num[0]}_{hkl_name_num[1]}_{hkl_name_num[2]}_gb"

    if stgb==False:
        dirname = f"{str(hkl_x)}_{str(hkl_y)}_{str(hkl_z)}"
        print("====================================================")
    else:
        pass
#    dirname = str(dirname)
    os.makedirs(dirname,exist_ok=True)
    os.chdir(dirname)
#     lattice_bi_copy = interface.lattice_bi.copy()
#     atoms_bi_copy = interface.atoms_bi.copy()
#     elements_bi_copy = interface.elements_bi.copy()
#     lattice_1_copy, atoms_1_copy, elements_1_copy = interface.lattice_1.copy(), interface.atoms_1.copy(), interface.elements_1.copy()
#     lattice_2_copy, atoms_2_copy, elements_2_copy = interface.lattice_2.copy(), interface.atoms_2.copy(), interface.elements_2.copy()  
    

    overwrite_supercell(interface,axis,n)

    
    if (axis_name == [1,1,1]):    
        interface.get_bicrystal(xyz_1 = [x,y,z], xyz_2 =[x,y,z],filename = 'atominfile', filetype='LAMMPS',mirror = True)
    else:
        interface.get_bicrystal(xyz_1 = [x,y,z], xyz_2 =[x,y,z],filename = 'atominfile', filetype='LAMMPS',mirror = False)


    surface.generate(os.getcwd())
    
# get surface atoms and mirrored surface atoms, 
# get atoms_surface_m(interface atoms)
    surface.get_surface()
# get atominfile2
    surface.get_atominfile2()
    
    CNID_hkl = dot(inv(interface.conv_lattice_1),interface.CNID)

    

    eps = 1e-5
    define_bicrystal_regions(interface.xhi)

    
    CNID = dot(interface.orient, interface.CNID)    
    length_1 = norm(CNID[:,0])
    length_2 = norm(CNID[:,1])
    area = norm(cross(CNID[:,0],CNID[:,1]))
    GB_area = norm(interface.lattice_bi[1])*norm(interface.lattice_bi[2])
    supercell_atoms = dot(interface.lattice_bi, interface.atoms_bi.T).T
    atoms_aroundgb = supercell_atoms[(supercell_atoms[:,0] >= (interface.xhi/2-bond_length*2)) & (supercell_atoms[:,0]<=(bond_length*2 + interface.xhi/2))]
    file.write('{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} \n'.format(CNID[:,0][1], CNID[:,0][2], \
                                                                   CNID[:,1][1], CNID[:,1][2], \
                                                                   length_1, length_2, area, sigma, axis_num, ab_num,GB_area, len(atoms_aroundgb)))

    v1 = np.array([0,1.,0])*CNID[:,0][1] + np.array([0,0,1.])*CNID[:,0][2]
    v2 = np.array([0,1.,0])*abs(CNID[:,1][1]) + np.array([0,0,1.])*abs(CNID[:,1][2])
    v2 = -v2


    
    n1 = int(ceil(norm(v1)/0.2))
    n2 = int(ceil(norm(v2)/0.2))
    write_trans_file(v1,v2,n1,n2)

    with open(f"{name}.meam","w") as f:
        f.writelines(element.meam())

    with open("library.meam","w") as f:
        f.writelines(element.library_meam())

    if stgb==False:
        with open("proto.in","w") as f:
            f.writelines(element.proto())
    else:
        pass


    os.chdir(os.pardir)

def define_bicrystal_regions(xhi):
    tol = 1e-5
    """
    generate a file defining some regions in the LAMMPS and define the atoms
    inside these regions into some groups.
    argument:
    region_names --- list of name of regions
    region_los --- list of the low bounds
    region_his --- list of the hi bounds
    """
    end_fixbulk1 = xhi/2-30
    start_fixbulk2 = xhi/2+30
    start_middle = xhi/2-20
    end_middle = xhi/2+20
    start_right = xhi/2 + tol
    start_bulk = 160
    end_bulk = 165


    with open('blockfile', 'w') as fb:
        fb.write('region fixbulk1 block EDGE {0:.16f} EDGE EDGE EDGE EDGE units box \n'.\
        format(end_fixbulk1))
        fb.write('region fixbulk2 block {0:.16f} EDGE EDGE EDGE EDGE EDGE units box \n'.\
        format(start_fixbulk2))
        fb.write('region middle block {0:.16f} {1:.16f} EDGE EDGE EDGE EDGE units box \n'.\
        format(start_middle,end_middle))
        fb.write('region right block {0:.16f} EDGE EDGE EDGE EDGE EDGE units box \n'.\
        format(start_right))
        fb.write('region bulk block {0:.16f} {1:.16f} EDGE EDGE EDGE EDGE units box \n'.\
        format(start_bulk,end_bulk))
        fb.write('group fixbulk1 region fixbulk1 \n')
        fb.write('group fixbulk2 region fixbulk2 \n')
        fb.write('group middle region middle \n')
        fb.write('group right region right \n')
        fb.write('group bulk region bulk \n')

def get_all_STGBs(axis_list, theta_list, sigma_list,name,element,max_sigma,stgb):
    eps = 1e-5
    file = open('CNIDs','w')
    cif_filename = glob.glob(f"C:/Users/hatay/calc/python_script/{name}/*.cif")[0]
    my_interface = core(cif_filename, cif_filename)
    my_interface.parse_limit(du = 1e-4, S  =  1e-4, sgm1=max_sigma, sgm2=max_sigma, dd =  1e-4)
    factor = element.getLatP()/(2*norm(my_interface.lattice_1[0,0]))
    my_interface.lattice_1 =  my_interface.lattice_1*factor
    my_interface.lattice_2 =  my_interface.lattice_2*factor
    my_interface.conv_lattice_1 =  my_interface.conv_lattice_1*factor
    my_interface.conv_lattice_2 =  my_interface.conv_lattice_2*factor  
    count = 1

    LatP = element.getLatP()
    bond_length = LatP/np.sqrt(2)
    coodinates = 12
    cut_off = 1.05    
    surface = Surface(LatP,bond_length,coodinates,cut_off)
    
    for i in range(len(axis_list)):
        axis = axis_list[i]
        axis_name = axis_list[i]
        axis_cart = dot(my_interface.conv_lattice_1, axis)

        axis = np.array(np.round(dot(inv(my_interface.lattice_1), axis_cart),8),dtype = int)
        for j in range(len(sigma_list[i])):

            sigma = sigma_list[i][j]
            my_interface.search_one_position(axis,theta_list[i][j]-0.001,1,0.001)
            CSL = my_interface.CSL
            n_1, n_2 = get_a_b(CSL, dot(my_interface.lattice_1,axis))

            hkl_1, hkl_2 = get_STGB_MLs(my_interface.lattice_1, n_1, n_2)

            hkl_name_1 = get_primitive_hkl(hkl_1, my_interface.lattice_1, my_interface.conv_lattice_1)
            hkl_name_2 = get_primitive_hkl(hkl_2, my_interface.lattice_2, my_interface.conv_lattice_2)
            
            get_gb_files(my_interface, hkl_1, axis, sigma, axis_name, hkl_name_1, 'a_{}'.format(count), file, i+1, 1, bond_length,n_1,surface,name,element,stgb)

            get_gb_files(my_interface, hkl_2, axis, sigma, axis_name, hkl_name_2, 'b_{}'.format(count+1), file, i+1, 2, bond_length,n_2,surface,name,element,stgb)

            count += 2
    file.close()

def get_theta_sigma_list(axis_list,max_sigma):
    theta_list = []
    sigma_list = []
    for i in axis_list:
        lists, thetas = getsigmas(i, max_sigma)
        theta_list.append(thetas)
        sigma_list.append(lists)
    return theta_list,sigma_list

# def read_atominfile(filename):
#     #read atom positions from POSCAR
#     print(filename)
#     with open (f"{filename}",'r') as f:
#         lines = f.readlines()
#     atoms = np.array([0,0,0,0,0])
#     for i in range(14,len(lines)):
#         atoms = np.vstack((atoms,np.array(lines[i].split(),dtype = float)))
#     atoms = np.delete(atoms,0,axis= 0)
#     atoms = np.delete(atoms, [0], 1)
#     a = float(lines[6].split()[1])
#     b = float(lines[7].split()[1])
#     c = float(lines[8].split()[1])
#     return a, b, c, atoms

def read_atominfile(dirname):
    #read atom positions from POSCAR
    print(f"{dirname}/atominfile")
    with open (f"{dirname}/atominfile",'r') as f:
        lines = f.readlines()
    atoms = np.array([0,0,0,0,0])
    for i in range(13,len(lines)):
        atoms = np.vstack((atoms,np.array(lines[i].split(),dtype = float)))
    atoms = np.delete(atoms,0,axis= 0)
    atoms = np.delete(atoms, [1], 1)
    a = float(lines[5].split()[1])
    b = float(lines[6].split()[1])
    c = float(lines[7].split()[1])
    return atoms, a, b, c

def read_atomin_id(filename):
    #read atom positions from POSCAR
    with open (f"{filename}",'r') as f:
        lines = f.readlines()
    atoms = np.array([0,0,0,0,0])
    for i in range(13,len(lines)):
        atoms = np.vstack((atoms,np.array(lines[i].split(),dtype = float)))
    atoms = np.delete(atoms,0,axis= 0)
    atoms = np.delete(atoms,[0],1)
    a = float(lines[5].split()[1])
    b = float(lines[6].split()[1])
    c = float(lines[7].split()[1])
    return a, b, c, atoms

def read_atomin_id2(dirname):
    #read atom positions from POSCAR
    with open (f"{dirname}/atominfile",'r') as f:
        lines = f.readlines()
    atoms = np.array([0,0,0,0,0])
    for i in range(13,len(lines)):
        atoms = np.vstack((atoms,np.array(lines[i].split(),dtype = float)))
    atoms = np.delete(atoms,0,axis= 0)
    a = float(lines[5].split()[1])
    b = float(lines[6].split()[1])
    c = float(lines[7].split()[1])
    return atoms

def Write_to_lammps(a,b,c,supercell_atoms,filename):
    dim = np.array([1,1,1])
    X = supercell_atoms.copy()

    NumberAt = len(X) 

    dimx, dimy, dimz = dim

    xlo = 0.00000000
    xhi = a
    ylo = 0.00000000
    yhi = b 
    zlo = 0.00000000
    zhi = c

    yz = 0.0

    Counter = np.arange(1, NumberAt + 1).reshape(1, -1)

    # data = np.concatenate((X_new, Y_new))
    FinalMat = np.concatenate((Counter.T, X), axis=1)

    with open(filename, 'w') as f:
        f.write('#Header \n \n')
        f.write('{} atoms \n \n'.format(NumberAt))
        f.write('2 atom types \n \n')
        f.write('{0:.8f} {1:.8f} xlo xhi \n'.format(xlo, xhi))
        f.write('{0:.8f} {1:.8f} ylo yhi \n'.format(ylo, yhi))
        f.write('{0:.8f} {1:.8f} zlo zhi \n\n'.format(zlo, zhi))
        f.write('{0:.8f} {1:.8f} {2:.8f} xy xz yz \n\n'.format(0, 0, yz))            
        f.write('Atoms \n \n')
        np.savetxt(f, FinalMat, fmt='%i %i %.8f %.8f %.8f')
    f.close()

def wrap_at_priodic_boundary_condition(b,c,atoms):
    tol = 0.01
    atoms[:, 2:4] = np.round(atoms[:,2:4],5)
    atoms[np.where(atoms[:,2] >(np.round(b,5)-tol)),2] = atoms[np.where(atoms[:,2] >(np.round(b,5)-tol)),2] - np.round(b,5)
    atoms[np.where(atoms[:,3] >(np.round(c,5)-tol)),3] = atoms[np.where(atoms[:,3] >(np.round(c,5)-tol)),3] - np.round(c,5)
    return atoms

# z方向に原子を反転する関数
def get_reverse(c,atoms):
    atoms[:,3] =c - atoms[:,3]

def check(v1,v2):
    """
    check v1 and v2 are orthogonal or parallel
    """
    rot_M = np.array([[0,-1,0],
                     [1,0,0],
                     [0,0,0]],dtype=int)
    v2_dash = np.dot(rot_M,v2.T).T
    if np.dot(v1,v2.T)==0 or np.dot(v1,v2_dash.T)==0:
        return_value=True
    else:
        return_value=False
    return return_value


def read_atominfile2(filename):
    #read atom positions from POSCAR
    with open (f"{filename}",'r') as f:
        lines = f.readlines()
    atoms = np.array([0,0,0,0,0])
    for i in range(14,len(lines)):
        atoms = np.vstack((atoms,np.array(lines[i].split(),dtype = float)))
    atoms = np.delete(atoms,0,axis= 0)
    atoms = np.delete(atoms, [0], 1)
    a = float(lines[6].split()[1])
    b = float(lines[7].split()[1])
    c = float(lines[8].split()[1])
    return a, b, c, atoms

def make_stgb(lattice_constant):
    for filename in glob.glob("*gb"):
        print(filename)
        filename_atominfile2 = f"./{filename}/atominfile2"
        filename_atominfile3 = f"./{filename}/atominfile3"
        gbname = np.array(filename.split("_")[:-1],dtype=int)
        print(gbname)
        a, b, c, atoms = read_atominfile2(filename_atominfile2)
        atoms_pbc = wrap_at_priodic_boundary_condition(b,c,atoms)
    #     gbname = np.array(dirname.split("_"),dtype=float)
        if gbname[1]/gbname[0]<1/2:
    #         print("this is under210")
    #         v_normal_to_plane = np.array(dirname.split("_")[3:6],dtype=int)
    #         v_normal_to_plane[0], v_normal_to_plane[1], v_normal_to_plane[2] \
    #         = v_normal_to_plane[1], v_normal_to_plane[2], v_normal_to_plane[0]
    #         v_gbindice = np.array(dirname.split("_")[0:3],dtype=int)
            v_normal_to_plane = gbname[3:6].copy()
            v_normal_to_plane[0], v_normal_to_plane[1], v_normal_to_plane[2] \
            = v_normal_to_plane[1], v_normal_to_plane[2], v_normal_to_plane[0]
            v_gbindice = gbname[0:3].copy()
            # checkを入れるところ
    #         print(v_gbindice)
    #         print(v_normal_to_plane)
            if check(v_gbindice,v_normal_to_plane):
                pass
            else:
                print("this is incorrect input\nneed mirror operation")
                get_reverse(c,atoms_pbc)
        elif gbname[1]/gbname[0]>1/2:
    #         print("this is over210")
    #         v_normal_to_plane = np.array(dirname.split("_")[3:6],dtype=int)
    #         v_normal_to_plane[0], v_normal_to_plane[1], v_normal_to_plane[2] \
    #         = v_normal_to_plane[1], v_normal_to_plane[2], v_normal_to_plane[0]
    #         v_gbindice = np.array(dirname.split("_")[0:3],dtype=int)
    #         v_gbindice[0], v_gbindice[1], v_gbindice[2] \
    #         = v_gbindice[1], v_gbindice[0], v_gbindice[2]
            v_normal_to_plane = gbname[3:6].copy()
            v_normal_to_plane[0], v_normal_to_plane[1], v_normal_to_plane[2] \
            = v_normal_to_plane[1], v_normal_to_plane[2], v_normal_to_plane[0]
            v_gbindice = gbname[0:3].copy()
            v_gbindice[0], v_gbindice[1], v_gbindice[2] \
            = v_gbindice[1], v_gbindice[0], v_gbindice[2]
            # checkを入れるところ
            if check(v_gbindice,v_normal_to_plane):
                pass
            else:
                print("this is incorrect input\nneed mirror operation")
                get_reverse(c,atoms_pbc)
        else:
            print("this is 210")
        center = a/2
        tol = 1e-5
        if float(gbname[1])/float(gbname[0]) < 1/2:
            # ここからが210よりも低角の粒界の凹凸を得るためのコード
            tilt_angle = np.arctan(float(gbname[1])/float(gbname[0]))
            top = (center - tol + (lattice_constant/2)*np.sin(tilt_angle))
            bottom = (center + tol - (lattice_constant/2)*np.sin(tilt_angle))
            atoms_need_modify = atoms_pbc[(bottom<atoms_pbc[:,1]) & (atoms_pbc[:,1]<top)]
            atoms_need_modify[:,1] = center
            reduced_atoms = np.unique(atoms_need_modify,axis=0)
            atoms_nocenter = atoms_pbc[(bottom>=atoms_pbc[:,1]) | (atoms_pbc[:,1]>=top)]
            atoms_final = np.vstack((atoms_nocenter,reduced_atoms))
            Write_to_lammps(a,b,c,atoms_final,filename_atominfile3)
        elif float(gbname[1])/float(gbname[0]) > 1/2:
            print(gbname)
            # ここから210よりも高角の粒界の凹凸を得るためのコード
            tilt_angle = np.arctan(float(gbname[0])/float(gbname[1]))
            theta = tilt_angle-(np.pi/4)
            top = (center - tol + (lattice_constant/2*np.sqrt(2))*np.sin(theta))
            bottom = (center + tol - (lattice_constant/2*np.sqrt(2))*np.sin(theta))
            atoms_need_modify = atoms_pbc[(bottom<atoms_pbc[:,1]) & (atoms_pbc[:,1]<top)]
            atoms_need_modify[:,1] = center
            reduced_atoms = np.unique(atoms_need_modify,axis=0)
            atoms_nocenter = atoms_pbc[(bottom>=atoms_pbc[:,1]) | (atoms_pbc[:,1]>=top)]
            atoms_final = np.vstack((atoms_nocenter,reduced_atoms))
            Write_to_lammps(a,b,c,atoms_final,filename_atominfile3)
        else:
            a, b, c, atoms = read_atominfile2(filename_atominfile2)
            Write_to_lammps(a,b,c,atoms,filename_atominfile3)
        new_dirname = "_".join(np.sort(gbname[0:3])[::-1].astype("str"))
        os.rename(filename,new_dirname)

class ProtoinCreator(metaclass=ABCMeta):
    def __init__(self,element,grand,mass):
        self.element = element
        self.grand = grand
        self.mass = mass
        self.cell = f"""
clear

#Initialize Simulation --------------------- 
units metal 
dimension 3 
boundary s p p
atom_style atomic
atom_modify map array
        """
        self.footer = f"""
#--------- RBT right atoms -------- 

thermo 10000 
thermo_style custom step lx ly lz c_emiddle temp c_hulk_dis_ave_x
dump            1 all custom 1 final id type x y z c_eng 
run 0

#4.excess energy
variable esum equal "v_minimumenergy * count(middle)" 
variable xseng equal "c_emiddle - (v_minimumenergy * count(middle))" 
variable gbe equal "(c_emiddle - (v_minimumenergy * count(middle)))/v_gbarea" 
variable gbemJm2 equal ${{gbe}}*16021.7733 
variable gbernd equal round(${{gbemJm2}}) 
variable ave_dis_x equal c_hulk_dis_ave_x


#----------- output calculation result of each loop into results file 
print "Grain Boundary energy (meV) = ${{gbemJm2}};"
print "All done!" 
        """
    
    def create(self,rbt):
        potential = self.potential()
        relaxtation = self.relaxation()
        compute_values = self.compute_values()
        cell = self.cell
        footer = self.footer
        inputfiles = self.inputfiles()
        return (f"{rbt}\n{self.cell}\n{inputfiles}\n{potential}\n{compute_values}\n{relaxtation}\n{footer}")
    
    def inputfiles(self):
        pass
    
    @abstractclassmethod
    def potential(self):
        pass

    @abstractclassmethod
    def relaxation(self):
        pass
    
    @abstractclassmethod
    def compute_values(self):
        pass

class atominfile3ProtoinCreator(ProtoinCreator):
    def potential(self):
        text = f"""
# ---------- Define Interatomic Potential --------------------- 
mass 1 {self.mass} #Cu
mass 2 {self.mass} #Cu

pair_style meam
pair_coeff * * library.meam {self.element} {self.element}.meam {self.element} {self.element}
neighbor 2.0 bin 
neigh_modify delay 10 check yes 
        """
        return (text)
    
    def relaxation(self):
        text = f"""
# ---------- Run Minimization ---------------------
reset_timestep 0

displace_atoms right move ${{tx}} ${{ty}} ${{tz}} units box

velocity fixbulk1 zero linear
fix fixbulk1 fixbulk1 setforce 0.0 0.0 0.0

velocity fixbulk2 zero linear
fix fixbulk2 fixbulk2 setforce NULL 0.0 0.0

min_style cg 
minimize 1.0e-10 1.0e-10 50000 100000 
        """
        return (text)
    
    def compute_values(self):
        text = f"""
# ---------- Compute properties of bulk --------------------- 
#0.excess energy
compute eng all pe/atom 
compute eatoms all reduce sum c_eng 
compute emiddle middle reduce sum c_eng
compute bulk_dis bulk displace/atom 
compute hulk_dis_ave_x bulk reduce ave c_bulk_dis[1]
# ---------- Calculate excess values ---------------------
#per atom properties in Cu crystal 
variable minimumenergy equal {self.grand}
variable gbarea equal "ly * lz" 
        """
        return text
    
    def inputfiles(self):
        text = """
# ---------- Create Atoms --------------------- 
read_data ./atominfile3
include ./blockfile
        """
        return text

def makeprotoin(lattice_constant,protoin,dx):
    alpha = np.arctan(1/2)
    for filename in glob.glob("*0"):
        gbname = filename.split("_")
        dy = lattice_constant/2
        keikaku = np.arctan(float(gbname[1])/float(gbname[0]))
        theta = alpha - keikaku
        gbcos = np.cos(theta)
        dz = -np.sqrt(5)*lattice_constant*gbcos/2
        rbt = f"""
variable tx equal {dx} # displacement in x direction
variable ty equal {dy} # displacement in y direction
variable tz equal {dz} # displacement in z direction

# end end of rbt parameter------------------
"""
        with open(f"./{filename}/proto.in","w") as f:
            f.write(protoin.create(rbt))


def main():
    print("input element name")
    name = input()
    print("max sigma")
    max_sigma = input()
    max_sigma = int(max_sigma)

    print("input dx value")
    dx = input()
    dx = float(dx)

    print("stgb?")
    stgb = input()

    stgb = bool(stgb)



    element = Element(name)
    protoin = atominfile3ProtoinCreator(element=name,grand=element.grand(),mass=element.mass())
    axis_list = [[1,0,0]]
    theta_list, sigma_list = get_theta_sigma_list(axis_list,max_sigma)
    
    
    get_all_STGBs(axis_list, theta_list, sigma_list, name, element,max_sigma,stgb)

    if bool(stgb):
        make_stgb(element.getLatP())
        makeprotoin(element.getLatP(),protoin,dx)
    else:
        pass



if __name__ == '__main__':
    main()