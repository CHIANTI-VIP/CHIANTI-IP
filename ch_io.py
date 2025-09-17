'''

This  program was originally part of the CHIANTI-VAMDC Interactive Python,
later renamed CHIANTI-IP, and part of CHIANTI Virtual IDL and Python (CHIANTI-VIP) 
software, an extension of the original CHIANTI software.


The original version was written in February 2016
within the UK-VAMDC consortium, with partial  funding (only for three months) by STFC (UK)
 by  Giulio Del Zanna (GDZ, University of Cambridge)


NAME: ch_io.py

PURPOSE: to read the CHIANTI database files and perform utility
         calculations. The codes follow as close as possible  those
         written  for the CHIANTI IDL software.

         *** Important***

        most of the functions and arrays have been given the same name as in the
        IDL codes, to make comparisons easier.
        The data and procedures are exactly the same as in the IDL codes, but some
        functionalities of Python and IDL are different, hence results are not exactly
        the same. A lot more work is needed to resolve the small differences!

Version:   1,  1 March 2016, GDZ 

Modified:

         modified some functions  to speed up the reading rather than using FOR loops.


         2,   4-Aug-2024, GDZ 



'''

import os, fnmatch
# import pickle
try:
    # for Python 3 import
    import configparser
except ImportError:
    # for Python 2 import
    import ConfigParser as configparser

import numpy as np
import pandas as pd
from io import StringIO

from scipy import interpolate

import fitsio

import pdb  # importing the python debugger   pdb.set_trace()


xuvtop=os.environ["XUVTOP"]

# default CHIANTI files
abund_file_default=os.path.join(xuvtop,'abundance','sun_photospheric_2021_asplund.abund')
ioneq_file_default=os.path.join(xuvtop,'ioneq','chianti.ioneq')


El = ['h','he','li','be','b','c','n','o','f','ne','na', \
    'mg','al','si','p','s','cl','ar','k','ca','sc','ti', \
    'v','cr','mn','fe','co','ni','cu','zn',\
    'ga','ge','as','se','br','kr']
Ionstage = ['I','II','III','IV','V','VI','VII','VIII','IX','X','XI','XII','XIII', \
    'XIV','XV','XVI','XVII','XVIII','XIX','XX','XXI',' XXII','XXIII','XXIV', \
    'XXV','XXVI','XXVII','XXVIII','XXIX','XXX','XXXI','XXXII','XXXIII','XXXIV', \
    'XXXV','XXXVI','XXXVII']
Spd = ['S', 'P', 'D', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'Q', 'R', 'T', 'U', 'V', 'W']


# some constants: 2022 CODATA Values. 

planck = 6.62607015e-27   #erg s
light = 29979245800.  # cm/s
planckEv = 4.135667696e-15  # ev s
boltzmann = 1.380649e-16  # cgs
ryd2Ev = 13.605693122     #Rydberg constant in eV
pi = 3.1415926535897931
ryd2erg = 2.179872361e-11  #Rydberg constant in erg
inv_cm_2erg = planck*light
emass = 9.10938215e-28  #  electron mass in gram
inv_cm_2ryd = 1./109737.32

# const_eie_rate produces the 8.63e-6 factor
const_eie_rate = planck**2/((2.*pi*emass)**1.5*np.sqrt(boltzmann))


def get_atomic_weight(zeta):

# Returns the average atomic weight, see 
# http://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl?ele=
    
    weights = [1.00794, 4.002602, 6.941, 9.0122, 10.811, 12.0107, 14.0067, 
               15.9994, 18.9984, 20.179, 22.98977, 24.3050, 26.9815, 28.086, 
               30.9738, 32.065, 35.453, 39.948, 39.0983, 40.078, 44.956, 
               47.90, 50.9414, 51.996, 54.9380, 55.845,
               58.9332, 58.6934, 63.546, 65.37]

    return weights[zeta-1]

def dilution(rsun):
    """
    Calculate the dilution factor: 

    rsun :  Distance from the center of the Sun in units of the radius.

    """
    if rsun >= 1.:
        d = 0.5*(1. - np.sqrt(1. - 1./rsun**2))
    else:
        d = 0.
    return d

def read_abundances(abundances_name=''):
    
    """
    reads a CHIANTI abundance file and returns the abundance values relative to hydrogen.

    """
    
    abundances=np.zeros((50),dtype="float64")

    if abundances_name!='':
        abundances_file=abundances_name
    else:
        # the user will select an abundances file

        fname = askopenfilename(title='Select an abundances file',
                                initialdir=os.path.join(xuvtop,'abundances'),
                                filetypes = (("Abundances files ","*.abund"),
                                                       ("All files", "*.*")))

        if fname == None:
            print((' no abundances file selected'))
            return 0
        else:
            abundances_file = fname # os.path.join(abundir, fname)

    input=open(abundances_file,'r')
    s1=input.readlines()
    input.close()
    nlines=0
    idx=-1
    while idx <= 0:
        minChar = min([5, len(s1[nlines])])
        aline=s1[nlines][0:minChar]
        idx=aline.find('-1')
        nlines+=1
    nlines-=1
    for line in range(nlines):
        z,ab,element=s1[line].split()
        abundances[int(z)-1]=float(ab)
    gz=np.nonzero(abundances)
    abs=10.**(abundances[gz]-abundances[0])
    abundances.put(gz,abs)

    abundances_Ref=s1[nlines+1:]

    return {'abundances_file':abundances_file,'abundances':abundances,'abundances_Ref':abundances_Ref}


def read_ioneq(ioneq_name='', verbose=0):
    """
    reads an ioneq file and stores temperatures and ionization equilibrium values in 
    a dictionary containing these value and the reference to the literature.
   
    """
    dir=os.environ["XUVTOP"]
    ioneqdir = os.path.join(dir,'ioneq')
    
    if ioneq_name == '':
        
        #  select an ioneq file
        fname1 = askopenfilename(initialdir= ioneqdir,
                                 filetypes = (("Ioneq files ","*.ioneq"),
                                              ("All files", "*.*")),
                                    title = 'Select an Ionization Equilibrium file')
        
        if fname == None:
            print(' no ioneq file selected')
            return False
        else:
            ioneqfilename=os.path.basename(fname)
            ioneq_name,ext=os.path.splitext(ioneqfilename)
    else:        
#check the input file exist !
        
        if not os.path.isfile(ioneq_name):
            print(('% read_ioneq:  input ioneq file does not exist, EXIT ! :  %s'%(ioneq_name)))
            return False
        else: fname=ioneq_name

    with open(fname, "r") as f:
        lines = f.readlines()

# Note: the number of temperatures can vary with the CHIANTI v.11.
        
    nt,nz=lines[0].split()
    nt=int(nt)
    nz=int(nz)

    logt=[float(x) for x in lines[1].split()]
    temperatures = 10.0**np.array(logt)
    
# read the temperatures:

    if verbose:
        print((' number of temperatures and elements = %5i %5i'%(nt, nz)))

    # find the index of the first "-1" line
    cut_index = None
    for i, line in enumerate(lines):
        if line.strip().startswith("-1"):
            cut_index = i
            break

# get the comments
    comm=lines[cut_index+1:]
# remove end of line charaters and empty strings: 
    comments=list(filter(None,  [element.strip() for element in comm]))

    # keep only lines before that
    if cut_index is not None:
        lines = lines[:cut_index]

# remove the first twqo lines:
        lines = lines[2:]
        
# number of data lines:        
    nlines=len(lines)
    
#hard-wired as in the IDL codes, also for version 11:
    ioneq=np.zeros((30,31, nt),dtype="float64")

# Not ideal solution to read line by line ...

    for il in range(nlines):

        line=lines[il]
        
        iz=int(line[0:3])
        ion=int(line[3:6])
        
        data=[]
        for j in range(nt):
            data.append(float(line[6+10*j:16+10*j]))

        ioneq[iz-1,ion-1,:]=data
            
    return {'ioneq_file':fname,'ioneq':ioneq,'temperatures':temperatures,'comments':comments}


# -------------------------------------------------------------------------------------

def p2eRatio(abundance_name, ioneq_name, eTemperature, eDensity):

    '''

    Calculates the proton  to electron density ratio.
    Uses the default elemental abundance file and ionization equilibrium file unless specified.

    '''

    if not abundance_name:
        abundance_name = abund_file_default

    abund=read_abundances(abundance_name)

    if not ioneq_name:
       ioneq_name= ioneq_file_default

    ioneq = read_ioneq(ioneq_name)

    temperature=np.asarray(eTemperature)
    nTemp=temperature.size
    eDensity=np.zeros(nTemp,dtype="float64")
    pDensity=np.zeros(nTemp,dtype="float64")
    ionDensity=0.

    #
    #  only hydrogen contributes to the proton density
    anEl = 0
    ion = 1
    good = ioneq['ioneq'][anEl,ion]  > 0.
    y2 = interpolate.splrep(np.log(ioneq['temperatures'][good]),
                            np.log(ioneq['ioneq'][anEl,ion,good]),s=0)
    bad1 = np.log(temperature) < np.log(ioneq['temperatures'][good].min())
    bad2 = np.log(temperature) > np.log(ioneq['temperatures'][good].max())
    bad=np.logical_or(bad1,bad2)
    goodt=np.logical_not(bad)
    thisIoneq=np.where(goodt,10.**interpolate.splev(np.log(temperature),y2,der=0),0.)
    pDensity+= abund['abundances'][anEl]*thisIoneq

    # all the rest do contribute to the electron and ion densities
    El=[iEl for iEl in range(50) if abund['abundances'][iEl] > 0.]
    for anEl in El:
        ionDensity+= abund['abundances'][anEl]
        for ion in range(1,anEl+2):
            good = ioneq['ioneq'][anEl,ion]  > 0.
            y2 = interpolate.splrep(np.log(ioneq['temperatures'][good]),
                                    np.log(ioneq['ioneq'][anEl,ion,good]),s=0)
            bad1 = np.log(temperature) < np.log(ioneq['temperatures'][good].min())
            bad2 = np.log(temperature) > np.log(ioneq['temperatures'][good].max())
            bad = np.logical_or(bad1,bad2)
            goodt = np.logical_not(bad)
            thisIoneq=np.where(goodt,10.**interpolate.splev(np.log(temperature),y2,der=0),1.)
            eDensity+=float(ion)* abund['abundances'][anEl]*thisIoneq

    ProtonDensityRatio=pDensity/eDensity
#        EDensity=eDensity
    IonDensity=ionDensity
    IonDensityRatio=ionDensity/eDensity

    return ProtonDensityRatio 


def convertname(name):
    """ convert CHIANTI ion name string to Z and Ion - same function as in IDL """
    s2=name.split('_')
    els=s2[0].strip()
    i1=El.index(els)+1
    ions=s2[1].strip()
    d=ions.find('d')
    if d >0 :
        dielectronic=True
        ions=ions.replace('d','')
    else: dielectronic=False
    higher = zion2name(int(i1), int(ions)+1)
    lower = zion2name(int(i1), int(ions)-1)
    return {'Z':int(i1),'Ion':int(ions),'Dielectronic':dielectronic, 'Element':els, 'higher':higher, 'lower':lower}
    #
    # -------------------------------------------------------------------------------------


def zion2filename(z,ion, dielectronic=False, xuvtop=0):
    
    """
    convert Z to generic file name string -  same function as in IDL

    """
    if xuvtop:
        dir = xuvtop
    else:
        dir=os.environ["XUVTOP"]
    if (z-1 < len(El)) and (ion <= z+1):
        thisel=El[z-1]
    else:
        thisel=''
    if z-1 < len(El):
        thisone=El[z-1]+'_'+str(ion)
        if dielectronic:
            thisone+='d'
    else:
        thisone=''
    if thisel != '' :
        fname=os.path.join(dir,thisel,thisone,thisone)
    return fname
 
    #
    # -------------------------------------------------------------------------------------
    #
def zion2spectroscopic(z,ion, dielectronic=False):
    """ 
    convert Z and ion to spectroscopic notation string - same as in IDL
    
    """
    if (z-1 < len(El)) and (ion <= z+1):
        spect=El[z-1].capitalize()+' '+Ionstage[ion-1]
        if dielectronic:
            spect+=' d'
    else:  spect = ''
    return spect
    #

def spectroscopic2name(el,roman):
    """ 
    convert Z and ion to spectroscopic notation string.
    Do some simple checking of the input.
    
    """
    elu = el.lower()
    romanu = roman.upper()
    # check the Ion stage first 
    try:
        idx = Ionstage.index(romanu)
    except ValueError:
        return ''
    #  check the element string

    try: 
        ind=El.index(elu)
    except ValueError:
        return ''

    gname = elu+'_'+str(idx+1)
    return gname

def zion2name(z,ion):
    """
    convert Z, ion to generic name. E.g.   26, 10 -> fe_10
    
    """
    if ion == 0:
        gname = ''
    elif ion == z+2:
        gname = ''
    elif (z-1 < len(El)) and (ion <= z+1):
        gname=El[z-1]+'_'+str(ion)        
    else:
        # this should not actually happen
        gname = ''
    return gname
    #
    # -------------------------------------------------------------------------------------


    
def read_elvlc(filename=0,  verbose=0  ):
    """
    reads the v. 8 CHIANTI format elvlc file
    the file that has 6  columns  and returns
    'lvl','conf','label','spin','spd','j','ecm','ecmth'

    Note: ecm is -1 when a state does not have an experimental energy.
    
    Written by G. Del Zanna
    
    """

# Define the column widths based on the Fortran format 'i7,a30,a5,i5,a5,f5.1,2f15.3'
# Note: np.genfromtxt requires the end index of each column
    col_widths = [(0, 7), (7, 37), (37, 42), (42, 47), (47, 52), (52, 57), (57, 72), (72, 87)]

# Define the data types for each column
# 'i' for integer, 'U' for Unicode string, 'f' for float
#    dtype = [int, 'U30', 'U5', int, 'U5', float, float, float]
    dtype = {0: int,1: str, 2: str,3: int, 4: str, 5: float,6: float, 7: float}

    #
    if not os.path.isfile(filename):
        print((' elvlc file does not exist, EXIT ! :  %s'%(filename)))
        return {'status':0}

    with open(filename, "r") as f:
     lines = f.readlines()

    # find the index of the first "-1" line
    cut_index = None
    for i, line in enumerate(lines):
        if line.strip().startswith("-1"):
            cut_index = i
            break

# get the comments
    comments=lines[cut_index+1:]
        
# keep only lines before that
    if cut_index is not None:
        lines = lines[:cut_index]
        
    data_str = "".join(lines)

    col_names = ['lvl','conf','label','spin','spd','j','ecm','ecmth']        

# *** note: the extra columns, if present, are not read in as in the IDL programs,
# as they can have different definitions.

    # data frame:
    df=pd.read_fwf(StringIO(data_str), colspecs=col_widths, dtype=dtype, names=col_names)

    data_dict = {col: df[col].to_numpy() for col in df.columns}
    # print(data_dict.keys())
        
    # remove end of line charaters and empty strings: 
    data_dict["comments"]= list(filter(None,  [element.strip() for element in comments]))
    data_dict['filename']=filename

    return data_dict

    #
    # -------------------------------------------------------------------------------------


def read_wgfa(filename=0,  verbose=0):
    """
    reads a chianti wgfa file and returns
    {"lvl1","lvl2","wvl","gf","avalue","comments"}
    
    
    """

    if not os.path.isfile(filename):
        print((' elvlc file does not exist, EXIT ! :  %s'%(filename)))
        return {'status':0}

    with open(filename, "r") as f:
        lines = f.readlines()

    # find the index of the first "-1" line
    cut_index = None
    for i, line in enumerate(lines):
        if line.strip().startswith("-1"):
            cut_index = i
            break

# get the comments
    comments=lines[cut_index+1:]
        
# keep only lines before that
    if cut_index is not None:
        lines = lines[:cut_index]
        
    data_str = "".join(lines)

    dtype = {0: int,1: int, 2: float,3: float, 4:float}
    col_names = ['lvl1','lvl2','wvl','gf','avalue']
    
    # equivalent to the FORTRAN Format='(2i5,f15.3,2e15.3)'
    col_widths = [(0, 5), (5, 10), (10, 25), (25, 40), (40, 55)]
    
# *** note: the extra columns, if present, are not read in as in the IDL programs,
# as they can have different definitions.

    # data frame:
    df=pd.read_fwf(StringIO(data_str), colspecs=col_widths, dtype=dtype, names=col_names)

    data_dict = {col: df[col].to_numpy() for col in df.columns}
    # print(data_dict.keys())

    # remove end of line charaters and empty strings: 
    data_dict["comments"]= list(filter(None,  [element.strip() for element in comments]))
    data_dict['filename']=filename
    
    return data_dict


def read_scups_fits(filename=0, verbose=0):

    '''
    to read the new format version 8 scups file in g-zipped FITS format
    containing the Burgess and Tully scaled temperature and upsilons.
    Written by G. Del Zanna
    
    '''
    #
        
    if not os.path.isfile(filename):
        print(('ERROR: g-zipped FITS  scups file does not exist:  %s'%(filename)))
        return {'status':0}
    
    data1,h = fitsio.read(filename, ext=1, header=True)
    data,h = fitsio.read(filename, ext=2, header=True)

#    pdb.set_trace()
    
    return {'lvl1':data['LVL1'], 'lvl2':data['LVL2'], 'ttype':data['T_TYPE'],'de':data['DE'], 'gf':data['GF'],
            'lim':data['LIM'], 'cups':data['C_UPS'],'ntemp':data['NSPL'],
            'btemp':data['STEMP'], 'bscups':data['SPL'], 'ntrans':len(data['LVL1']), 'ref':data1['COMMENTS']}
    
    # Read the proton excitation data in the .pslups files
    # --------------------------------------------------
    #
    
def read_proton_splups(filename=0):

    """
    read a chianti proton splups file and return
    
    {"lvl1","lvl2","ttype","gf","de","cups","bsplups", "comments"}

    **** Note: the number of spline points depends on the original file.
         Most CHIANTI files have either 5 or 9. We enforce a maximum of 9 here.

     """

    if not os.path.isfile(filename):
        print((' proton splups file does not exist, EXIT ! :  %s'%(filename)))
        return {'status':0}

    with open(filename, "r") as f:
        lines = f.readlines()

    # find the index of the first "-1" line
    cut_index = None
    for i, line in enumerate(lines):
        if line.strip().startswith("-1"):
            cut_index = i
            break

# get the comments
    comments=lines[cut_index+1:]
    # remove end of line charaters and empty strings: 
    comments_r= list(filter(None,  [element.strip() for element in comments]))

    
# keep only lines before that
    if cut_index is not None:
        lines = lines[:cut_index]

# number of data lines:        
    nlines=len(lines)
    
# now we need to find out how many spline nodes there are. At most 9 !
# the first numbers are lvl1, lvl2,  t_type, gf, de , c_ups with IDL format='(3i3,3e10.3)'

    nchar = len(lines[0])
# need to round as in IDL you get an integer, but in Python a real.    
    nspl = round((nchar - 3*3 - 3*10)/10 ) 
    if nspl > 9:
        nspl = 9

    lvl1=[] ; lvl2=[] ; t_type=[] ; gf=[] ; de=[] ;c_ups=[]

    spl=np.zeros((nspl, nlines),dtype="float64") ;    spl[:]=-1.
        
    for il in range(nlines):

        line=lines[il]
        lvl1.append(int(line[0:3]))
        lvl2.append(int(line[3:6]))
        t_type.append(int(line[6:9]))
        gf.append(float(line[9:19]))
        de.append(float(line[19:29]))
        c_ups.append(float(line[29:39]))

        data=[]
        for j in range(nspl):
            data.append(float(line[39+10*j:49+10*j]))

        spl[0:nspl, il]=data
            
    return {"lvl1":lvl1,"lvl2":lvl2,"t_type":t_type,"gf":gf,"de":de,"c_ups":c_ups
                ,"nspl":nspl,"spl":spl,"comments":comments_r, 'filename':filename}


#-----------------------------------------------------------


def descale_proton_splups(proton_spl, temperature):

        """
        Provides the rate coefficients from the proton scaled data
        Input: the scaled data  and the temperatures
        
        Output: the descaled rates upsilon, which are not, as in the case of the 
        electrons, the effective collision strengths but rather the rates coefficients.

        *** Note: the number of spline points varies depending on the original file.
        
        """

        if type(proton_spl) == type(None):
            PUpsilon = None
            print(' Proton rates undefined')
            return {'errorMessage':' Proton rates undefined'}
        else:
            nsplups = len(proton_spl["lvl1"])

        if type(temperature) == type(None):
            print(' Temperature undefined')
            return {'errorMessage':' Temperature undefined'}

        temp=np.asarray(temperature)
        ntemp=temp.size
        if ntemp > 1:
            ups = np.zeros((nsplups,ntemp), dtype="float64")
        else:
            ups = np.zeros(nsplups,dtype="float64")

        #
        nspl=proton_spl["nspl"]
        dx=1./(float(nspl)-1.)

#        pdb.set_trace()
        
        for isplups in range(nsplups):
   
            # for proton rates
            l1=proton_spl["lvl1"][isplups]-1
            l2=proton_spl["lvl2"][isplups]-1
            t_type=proton_spl["t_type"][isplups]
            c_ups=proton_spl["c_ups"][isplups]
            splups=proton_spl["spl"][:,isplups]
            de=proton_spl["de"][isplups]

            kte = boltzmann*temp/(de*ryd2erg)

            der=0
            if t_type == 1:
                st=1.-np.log(c_ups)/np.log(kte+c_ups)
                xs=dx*np.arange(nspl)
                y2=interpolate.splrep(xs,splups,s=0)
                sups=interpolate.splev(st,y2,der=der)
#                sups=interpolate.spline(xs, splups, st)
                ups[isplups]=sups*np.log(kte+np.exp(1.))
            #
            if t_type == 2:
                
                st=kte/(kte+c_ups)
                xs=dx*np.arange(nspl)
                y2=interpolate.splrep(xs,splups,s=0)
                sups=interpolate.splev(st,y2,der=der)
                ups[isplups]=sups
            #
            if t_type == 3:
                st=kte/(kte+c_ups)
                xs=dx*np.arange(nspl)
                y2=interpolate.splrep(xs,splups,s=0)
                sups=interpolate.splev(st,y2,der=der)
                ups[isplups]=sups/(kte+1.)
            #
            if t_type == 4:
                st=1.-np.log(c_ups)/np.log(kte+c_ups)
                xs=dx*np.arange(nspl)
                y2=interpolate.splrep(xs,splups,s=0)
                sups=interpolate.splev(st,y2,der=der)
                ups[isplups]=sups*np.log(kte+c_ups)
            #
            if t_type == 5:
                # dielectronic rates
                st=kte/(kte+c_ups)
                xs=dx*np.arange(nspl)
                y2=interpolate.splrep(xs,splups,s=0)
                sups=interpolate.splev(st,y2,der=der)
                ups[isplups]=sups/(kte+0.)
            #
            #  descale proton values
            if t_type == 6:
                st=kte/(kte+c_ups)
                xs=dx*np.arange(nspl)
                y2=interpolate.splrep(xs,splups,s=1.e-5)
                sups=interpolate.splev(st,y2,der=der)
                ups[isplups]=10.**sups
            #
            elif t_type > 6:  print(' t_type ne 1,2,3,4,5 = %5i %5i %5i'%(t_type,l1,l2))            
            
        #
        ups=np.where(ups > 0.,ups,0.)

        output = {'rate_coeff':ups }
        return output


def descale_scups(scups_data, temperature):

    """
    Calculates the effective collision strengths (upsilon)
    for electron excitation as a function of temperature.

    Same as the IDL program.

    """
    #
    #  xt=kt/de
    #
    #  need to make sure elvl is >0, except for ground level
    ntemp=temperature.size
    nsplups=len(scups_data['de'])
    if ntemp > 1:
        ups=np.zeros((nsplups,ntemp),dtype="float64")
    else:
        ups=np.zeros(nsplups,dtype="float64")
    #
    for iscups in range(0,nsplups):

        l1=scups_data["lvl1"][iscups]-1
        l2=scups_data["lvl2"][iscups]-1
        ttype=scups_data["ttype"][iscups]
        cups=scups_data["cups"][iscups]
        nspl=scups_data["ntemp"][iscups]    
        xs=scups_data["btemp"][iscups]
        scups=scups_data["bscups"][iscups]
        de=scups_data["de"][iscups]
        kte = boltzmann*temperature/(de*ryd2erg)

        #
        der=0
        if ttype == 1:
            st=1.-np.log(cups)/np.log(kte+cups)
            y2=interpolate.splrep(xs,scups,s=0)
            sups=interpolate.splev(st,y2,der=der)
            ups[iscups]=sups*np.log(kte+np.exp(1.))
        #
        if ttype == 2:
            st=kte/(kte+cups)
            y2=interpolate.splrep(xs,scups,s=0)
            sups=interpolate.splev(st,y2,der=der)
            ups[iscups]=sups
        #
        if ttype == 3:
            st=kte/(kte+cups)
            y2=interpolate.splrep(xs,scups,s=0)
            sups=interpolate.splev(st,y2,der=der)
            ups[iscups]=sups/(kte+1.)
        #
        if ttype == 4:
            st=1.-np.log(cups)/np.log(kte+cups)
            y2=interpolate.splrep(xs,scups,s=0)
            sups=interpolate.splev(st,y2,der=der)
            ups[iscups]=sups*np.log(kte+cups)

        elif ttype > 5:  print((' t_type ne 1,2,3,4,5 = %5i %5i %5i'%(ttype,l1,l2)))
    #
    #
    ups=np.where(ups > 0.,ups,0.)
    #
    return ups



