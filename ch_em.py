'''

This  program was originally part of the CHIANTI-VAMDC Interactive Python,
later renamed CHIANTI-IP, and part of CHIANTI Virtual IDL and Python (CHIANTI-VIP) 
software, an extension of the original CHIANTI software.


The original version was written in February 2016
within the UK-VAMDC consortium, with partial  funding (only for three months) by STFC (UK)
 by  Giulio Del Zanna (GDZ, University of Cambridge)

NAME: ch_em.py


PURPOSE: to calculate the ion emissivities, as close as possible to those
         obtained with the CHIANTI IDL software.

         *** Important***

        most of the functions and arrays have been given the same name as in the
        IDL codes, to make comparisons easier.
        The data and procedures are exactly the same as in the IDL codes, but some
        functionalities of Python and IDL are different, hence results are not exactly
        the same.
        A lot more work is needed to resolve the small differences!

        Also, not all the functionalities and options within the IDL codes have been ported.
        

		hc
        	--   N_j   A_ji
      	     4 pi  lambda

  where h is Planck's constant, c speed of light,
  lambda is the wavelength in angstroms, N_j is the relative
  abundance of the upper state j (normalized so the total population
  of all the states is 1),  and A_ji is Einstein's 
  radiative decay rate for the transition to the lower level i.

  so the Units are erg s-1 sr-1

  Note: to get the standard emissivity or contribution function of a transition
  in erg cm+3 s-1 sr-1
  one needs to divide this output for the electron density, multiply it
  for the ion abundance (relative fraction of the ion abundance),
  the elemental abundance relative to hydrogen,
  and the hydrogen abundance relative to the electron number density,
  which is about 0.83 for a coronal plasma with H,He fully ionized.

  For details see e.g. Del Zanna & Mason, Living Review, 2018.



INPUTS: 
       ion_string
       the ion in CHIANTI format (e.g., 'fe_10')

	temperature
        the electron temperature array [K]

	density
        the electron density array [cm-3]

OPTIONAL INPUTS:

        radTemperature
        the stellar  radiation temperature 

         Star
          Distance from the centre of the star in stellar radius units.

         proton_rates
         to add collisional proton rates (if available)

OUTPUT:
         Emiss['ion']
          Ion

         Emiss['lvl1'], Emiss['lvl2'] 
           lower and upper CHIANTI level number

         Emiss['conf1'],Emiss['conf2']
         lower, upper state configuration

         Emiss['spin1'],Emiss['spin2']
         lower, upper 2S+1

         Emiss['spd1'],Emiss['spd2']
         lower, upper L

         Emiss['j1'],Emiss['j2']
         lower, upper J

         Emiss['em']
         emissivities

         Emiss['wvl']
         wavelengths in Angstroms

         Emiss['avalue']
         Einstein's A-values 


Version:   1,  1 March 2016, GDZ 

Modified:
         added photo-excitation and modified some functions, mainly
         reading the scups.fits.gz files to speed up.

         2,   4-Aug-2024, GDZ 

'''


import os
import copy
import numpy as np
from scipy import interpolate

# import the CHIANTI-IP reading routines 
import ch_io

import pdb  # importing the python debugger   pdb.set_trace()
import warnings


class emiss(object):

    '''
    The top level class for getting the emissivity of an ion in the CHIANTI database.

    ion_string is a string corresponding such as 'c_5' that corresponds to the C V ion.
    temperature in Kelvin
    eDensity in cm^-3
    radTemperature, the radiation black-body temperature in Kelvin
    rStar, the distance from the center of the star in stellar radii
            
    '''
    def __init__(self, ion_string, eTemperature=None, eDensity=None, protonDensity =None, 
                 radTemperature=0,  rStar=0, abundanceName=0, abundance=0,  verbose=0, 
                 setup=True, em=0,  proton_rates=0,wmin=None, wmax=None):

        self.IonStr=ion_string
        self.Z=ch_io.convertname(ion_string)['Z']
        self.Ion=ch_io.convertname(ion_string)['Ion']
        self.Spectroscopic=ch_io.zion2spectroscopic(self.Z,self.Ion)
        FileName=ch_io.zion2filename(self.Z, self.Ion )
        self.FileName=FileName

        self.pDensity=protonDensity
        self.proton_rates=proton_rates

        self.eTemperature=eTemperature
        self.eDensity=eDensity

        self.RadTemperature =radTemperature
        self.RStar =rStar

        # run the emissivity calculation:
        self.emiss_calc(wmin=wmin, wmax=wmax)        

         
        #
        # ------------------------------------------------------------------------------
        #
    def emiss_calc(self, wmin, wmax,  allLines=1):
        """
        Calculate the emissivities for lines of the specified ion.

        wvlRange can be set to limit the calculation to a particular wavelength range

        units:  erg  s-1 sr-1

        Does not include elemental abundance or ionization fraction

        Wavelengths are sorted
        set allLines = 1 to include unidentified lines
        """

        self.Wgfa = ch_io.read_wgfa(filename=self.FileName+'.wgfa')

        #
        wvl = np.asarray(self.Wgfa["wvl"], 'float64')
        obs = np.where(wvl > 0., 'Y', 'N') # numpy.ndarray 
        if allLines:
            wvl=np.abs(wvl)
        l1 = np.asarray(self.Wgfa['lvl1'], 'int64')
        l2 = np.asarray(self.Wgfa["lvl2"], 'int64')
        avalue = np.asarray(self.Wgfa["avalue"], 'float64')

        #
        # make sure there are lines in the wavelength range, if specified        
        
        if wmin or wmax:
            good = np.array(np.where((wvl >= wmin) & (wvl <= wmax)))
            l1 = l1[good]
            l2 = l2[good]
            wvl = wvl[good]
            avalue = avalue[good]
            obs = obs[good]

        # two-photon decays have wvl=0 and nonzero avalues
        nonzed = wvl != 0.
        wvl = wvl[nonzed]
        l1 = l1[nonzed]
        l2 = l2[nonzed]
        avalue = avalue[nonzed]
        obs = obs[nonzed]
        nwvl=len(wvl)

        #
        if nwvl == 0:
            self.warnings="Warning! no lines in this wavelength range "
            return 

 
        self.Nwgfa=len(self.Wgfa['lvl1'])
        nlvlWgfa = max(self.Wgfa['lvl2'])
        nlvlList =[nlvlWgfa]

        Elvlc = ch_io.read_elvlc(filename=self.FileName+'.elvlc')

        # now for what follows we need to remove the -1 from the energies and add extra info
        # the problem here is that energy ordering might be messy
        energies_cm=Elvlc['ecm']
        index_bad=np.array(np.where(energies_cm == -1))
        if index_bad.size > 1:
            energies_cm[index_bad]=Elvlc['ecmth'][index_bad]


        Elvlc['energies_cm']=energies_cm
        Elvlc["energies_ryd"]=energies_cm* ch_io.inv_cm_2ryd
        Elvlc["mult"]=(2.* Elvlc['j'])+1.

        
        self.Elvlc=Elvlc
        
        # needed to determine the number of levels that can be populated
        nlvlElvlc = len(self.Elvlc['lvl'])
        #  elvlc file can have more levels than the rate level files
        self.Nlvls = min([nlvlElvlc, max(nlvlList)])


        self.Scups = ch_io.read_scups_fits(filename=self.FileName+'.scups.fits.gz')

        self.Nscups=len(self.Scups['lvl1'])
        nlvlScups = max(self.Scups['lvl2'])
        nlvlList.append(nlvlScups)

        #  read proton rates:
        if self.proton_rates:

            #  psplups file may not exist
            psplupsfile = self.FileName +'.psplups'
            if os.path.isfile(psplupsfile):
                self.proton_spl = ch_io.read_proton_splups(filename=psplupsfile)
                self.Npsplups=len(self.proton_spl["lvl1"])
                verbose=1
                if verbose: 
                    print('we have read the proton file: ',psplupsfile)
            else:
                self.Npsplups = 0
        else:
            self.Npsplups = 0

        eTemperature=self.eTemperature
        eDensity=self.eDensity

        self.populations(eTemperature, eDensity)
        pop=self.Population["pop"]
        

        try:
            ntempden,nlvls = pop.shape
            em=np.zeros((nwvl, ntempden),dtype="float64")
            if np.asarray(self.eDensity).size < ntempden:
                eDensity = np.repeat(np.asarray(self.eDensity), ntempden)
            else:
                eDensity = np.asarray(self.eDensity)
        except:
            ntempden=1
            em=np.zeros(nwvl,dtype="float64")
            eDensity = np.asarray(self.eDensity)

        # set default to ergs 
        factor=ch_io.planck*ch_io.light/(4.*ch_io.pi*1.e-8*wvl)
        ylabel="erg  s-1 sr-1"

        if ntempden > 1:
            for itempden in range(ntempden):
                for iwvl in range(nwvl):
                    p = pop[itempden,l2[iwvl]-1]
                    em[iwvl, itempden] = factor[iwvl]*p*avalue[iwvl]
        else:
            for iwvl in range(0,nwvl):
                p=pop[l2[iwvl]-1]
                em[iwvl]=factor[iwvl]*p*avalue[iwvl]

        
        nlvl = len(l1)
        ion = np.asarray([self.IonStr]*nlvl)
        
        Emiss =  np.zeros(nlvl, dtype=[('lvl1','i4'),('lvl2','i4'),('ion','a5'),
                                       ('em','f8',ntempden),('wvl','f8'),('avalue','f8'),
                                       ('conf1', 'a30'), ('conf2', 'a30'), 
                                       ('spin1', 'i4'), ('spin2', 'i4'),
                                       ('spd1', 'a3'), ('spd2', 'a3'), 
                                       ('j1', 'f4'), ('j2', 'f4'), ])
        
        Emiss['lvl1']=l1  
        Emiss['lvl2']=l2
        Emiss['ion']=ion
        Emiss['em']= em
        Emiss['wvl']=wvl
        Emiss['avalue']=avalue

        # add spectroscopic notation: 
        ind1=np.zeros(nwvl, dtype=np.int64) ; ind2=np.zeros(nwvl, dtype=np.int64) 

        for il in range(0,nwvl):
            
            ii1=np.where(self.Elvlc['lvl'] == l1[il])
            ind1[il]=ii1[0]            
            ii2=np.where(self.Elvlc['lvl'] == l2[il])        
            ind2[il]=ii2[0]
            
        Emiss['conf1']= np.asarray(self.Elvlc['conf'])[ind1]
        Emiss['conf2']= np.asarray(self.Elvlc['conf'])[ind2] 
        Emiss['spin1']= np.asarray(self.Elvlc['spin'])[ind1] 
        Emiss['spin2']= np.asarray(self.Elvlc['spin'])[ind2] 
        Emiss['spd1']= np.asarray(self.Elvlc['spd'])[ind1] 
        Emiss['spd2']=np.asarray(self.Elvlc['spd'])[ind2] 
        Emiss['j1']= np.asarray(self.Elvlc['j'])[ind1] 
        Emiss['j2']= np.asarray(self.Elvlc['j'])[ind2] 

        self.warnings=''
        self.Emiss = Emiss
        return

    # -------------------------------------------------------------------------------------
    #


    def populations(self, popCorrect=0, verbose=0, **kwargs ):
        """

        Calculate level populations for specified ion.
        possible keyword arguments include eTemperature, eDensity,  radTemperature and rStar
        eTemperature and eDensity need to be defined  !

        this function follows closely the IDL programs, giving the same names to the
        various arrays.

        
        """

        temperature=np.asarray(self.eTemperature)
        ntemp=temperature.size

        density=np.asarray(self.eDensity)
        ndens = density.size


        if ntemp > 1 and ndens >1 and ntemp != ndens:
            print(' unless temperature or density are single values')
            print(' the number of temperatures values must match ')
            print(' the number of density values')
            return {'errorMessage': 'error in ch_em.populations ! number of temperatures must match the number of densities' }


        if  hasattr(self, 'PDensity'):
            if type(self.pDensity) == type(None):

                self.ProtonDensityRatio=ch_io.p2eRatio(ch_io.abund_file_default, 
                                                       ch_io.ioneq_file_default,
                                                       self.eTemperature, self.eDensity)
                self.pDensity = self.ProtonDensityRatio*density
                protonDensity = self.pDensity

            else: protonDensity = self.pDensity
        else:
            self.ProtonDensityRatio=ch_io.p2eRatio(ch_io.abund_file_default, 
                                                   ch_io.ioneq_file_default,
                                                   self.eTemperature, self.eDensity)
            self.pDensity = self.ProtonDensityRatio*density
            protonDensity = self.pDensity

        # these values have been set already when reading the files.
        n_levels=self.Nlvls
        nwgfa=self.Nwgfa
        nscups=self.Nscups
        npsplups=self.Npsplups
        #

# GDZ: for now DO NOT apply the corrections to level populations due to recombination. 
# They are minor corrections and the way is done in CHIANTI is a rough approximation which is not correct.

        ci=0
        rec=0

        if  self.RadTemperature > 0 : 
                dilution = ch_io.dilution(self.RStar)

        # this is the populating matrix for radiative transitions
        aa=np.zeros((n_levels,n_levels),dtype="float64")
        
        # this is the matrix for photo-excitation
        pexc=np.zeros((n_levels,n_levels),dtype="float64")
        # this is the matrix for stimulated emission
        stem=np.zeros((n_levels,n_levels),dtype="float64")

        result=np.zeros((n_levels,n_levels),dtype="float64")
        mm=np.zeros((n_levels,n_levels),dtype="float64")
        
        for iwgfa in range(nwgfa):

            l1 = self.Wgfa["lvl1"][iwgfa]-1
            l2 = self.Wgfa["lvl2"][iwgfa]-1
#            rad[l1+ci,l2+ci] += self.Wgfa["avalue"][iwgfa]
            aa[l2,l1] += self.Wgfa["avalue"][iwgfa]

            
#   photo-excitation and stimulated emission for a black-body:

            if  self.RadTemperature > 0 : 

                # do not include autoionization states:
                if abs(self.Wgfa['wvl'][iwgfa]) > 0.:

                    #print ('adding PE for ')
                    #print (self.Wgfa['wvl'][iwgfa])
                    
                    dd = ch_io.inv_cm_2erg/ch_io.boltzmann*abs(self.Elvlc['energies_cm'][l2] - self.Elvlc['energies_cm'][l1]) \
                    / self.RadTemperature 
#; the following lines are necessary to prevent infinities and floating underflow errors in IDL.
# WE FOLLOW HERE THE SAME coding as in IDL.

                    #   wrong syntax for python:   dd=dd < 150.
                    
                    if dd == 0.:
                        ede=1e50 #; arbitrarily large value
                        result[l1,l2]=0.0 
                    elif dd >=  1e-15 and dd < 150. :
                        ede=np.exp(dd) - 1.
                        result[l1,l2]=1.0/ede
                    elif dd < 1e-15:
                        ede=dd
                        result[l1,l2]=1.0/ede
                    elif dd >= 150:
                        ede=np.exp(150.) - 1.
                        result[l1,l2]=1.0/ede

                    mm[l1,l2] =float(self.Elvlc['mult'][l2])/float(self.Elvlc['mult'][l1])

# photo-excitation matrix:                    
                    pexc[l1,l2] = self.Wgfa["avalue"][iwgfa]*dilution* mm[l1,l2]*result[l1,l2]
                    
# stimulated emission matrix:

                    stem[l2, l1] = self.Wgfa["avalue"][iwgfa]*dilution*result[l1,l2]


# sum the radiative matrices:

        rad=np.array(aa+pexc+stem)             
        self.rad=rad


# as in the IDL code make the qq array for electron rates:
        qq=np.zeros((ntemp,n_levels,n_levels),dtype="float64")
        

# use the descaling routine in ch_io .  Do all temperatures at once
#  ups is  an array (nscups,ntemp) or (nscups) for one temperature case *** 
        ups=ch_io.descale_scups(self.Scups, temperature)

        
        for iscups in range(nscups):
            
            l1=self.Scups["lvl1"][iscups]-1
            l1idx = np.where(self.Elvlc['lvl'] ==(self.Scups['lvl1'][iscups]))
            l2=self.Scups["lvl2"][iscups]-1
            l2idx = np.where(self.Elvlc['lvl'] ==(self.Scups['lvl2'][iscups]))

            # remove any states above n_levels
            if l1 <= n_levels and l2 <= n_levels:

                # avoid negative values - in IDL we do a different check.
                # difference in energy in Rydberg:
                de = np.abs(self.Elvlc["energies_ryd"][l2idx] - self.Elvlc["energies_ryd"][l1idx])
                ekt = (de*ch_io.ryd2erg)/(ch_io.boltzmann*temperature)
                fmult1 = float(self.Elvlc["mult"][l1idx])
                fmult2 = float(self.Elvlc["mult"][l2idx])

                #excitation -  ch_io.const_eie_rate= 8.62913438e-6                
                qq[:, l1, l2]= ch_io.const_eie_rate *ups[iscups]*np.exp(-ekt)/(fmult1*np.sqrt(temperature))
                # de-excitation:
                qq[:, l2, l1]= ch_io.const_eie_rate*ups[iscups]/(fmult2*np.sqrt(temperature))


        ppr=np.zeros((ntemp,n_levels,n_levels),dtype="float64")
        #  add proton excitation and de-excitation 
        if npsplups > 0:

            
            proton_rate_c=ch_io.descale_proton_splups(self.proton_spl, temperature)

            for isplups in range(npsplups):
                
                l1=self.proton_spl["lvl1"][isplups]-1
                l2=self.proton_spl["lvl2"][isplups]-1
                fmult1 = float(self.Elvlc["mult"][l1])
                fmult2 = float(self.Elvlc["mult"][l2])

                # avoid negative values - in IDL we do a different check on the energies. take the absolute?
                # chiantiPy did not take the absolute
                de=abs(self.proton_spl["de"][isplups])

# the original version of      ChiantiPy had the proton rates completely wrong.           
# ChiantiPy did not multiply here for the pe_ratio, it does later multiply for self.pDensity

                # excitation:
                ppr[:, l1, l2]=proton_rate_c["rate_coeff"][isplups]*self.ProtonDensityRatio 
                ppr[:, l2, l1]=proton_rate_c["rate_coeff"][isplups]*self.ProtonDensityRatio * \
                    fmult1 / fmult2 *  np.exp(de*13.61/8.617/10.**(-5.)/temperature)

 
        if ntemp == 1 and ndens > 1:

            # add A-values and photo-excitation, plus de-xcitation processes:
            #c_noncoll= rad
            #Add all the rate coefficients for the processes that are
            #proportional to the electron density:
            #c_coll= qq[0,:,:] + ppr[0,:,:]
            
            
            pop=np.zeros((ndens,n_levels),dtype="float64")
            
            for idens in range(0,ndens):
                
                scalar=density[idens]                
                C= rad + scalar.item()* ( qq[0,:,:] + ppr[0,:,:])

                diag = -np.sum(C, axis=1)
                np.fill_diagonal(C, diag)
# Zero the first column and set C[0,0] to 1.0
                C[:,0]=0.0
                C[0,0]=1.0
                C_to_solve = C.T

# Solve the system (C_to_solve) * x = b1 using np.linalg.solve
# This is equivalent to solving:
# (Matrix A after diagonal fill, zero first column, C[0,0]=1.0).T * x = b

# Note: this is equivalent to the original way it was coded in IDL,
# and in most cases gives exactly the same answer as the IDL v.11 ,
# but more work needs to carried out to resolve any differences. This is NOT trivial!

                b=np.zeros(n_levels,dtype="float64")
                b[0]=1.
                x=np.linalg.solve(C_to_solve, b)
                
                pop[idens,:] = x/np.sum(x)

        elif ndens == 1 and ntemp > 1 :

            pop=np.zeros((ntemp, n_levels),"float64")           
            
            for itemp in range(0,ntemp):
                              
                C= rad +  density* ( qq[itemp,:,:] + ppr[itemp,:,:])

                diag = -np.sum(C, axis=1)
                np.fill_diagonal(C, diag)
                C[:,0]=0.0
                C[0,0]=1.0
                C_to_solve = C.T
                b=np.zeros(n_levels,dtype="float64")
                b[0]=1.
                x=np.linalg.solve(C_to_solve, b)
                pop[itemp,:] = x/np.sum(x)

            
        elif ntemp == 1 and ndens == 1:

            pop=np.zeros(n_levels)

            C= rad + density * ( qq[0,:,:] + ppr[0,:,:])
            diag = -np.sum(C, axis=1)
            np.fill_diagonal(C, diag)
            C[:,0]=0.0
            C[0,0]=1.0
            C_to_solve = C.T
            b=np.zeros(n_levels,dtype="float64")
            b[0]=1.
            x=np.linalg.solve(C_to_solve, b)
            pop[idens,:] = x/np.sum(x)
 
        elif ntemp > 1  and ntemp==ndens:
            
            pop=np.zeros((ntemp,n_levels),dtype="float64")
            
            for itemp in range(0,ntemp):

                scalar=density[itemp]                
                C= rad + scalar.item()* ( qq[itemp,:,:] + ppr[itemp,:,:])

                diag = -np.sum(C, axis=1)
                np.fill_diagonal(C, diag)
                C[:,0]=0.0
                C[0,0]=1.0
                C_to_solve = C.T
                b=np.zeros(n_levels,dtype="float64")
                b[0]=1.
                x=np.linalg.solve(C_to_solve, b)
                pop[itemp,:] = x/np.sum(x)
            

        pop=np.where(pop >0., pop,0.)
        self.Population={"eTemperature":self.eTemperature,"eDensity":self.eDensity,
                         "pop":pop, "protonDensity":protonDensity, "ci":ci, "rec":rec}
        #
        return

   #
    # -------------------------------------------------------------------------------------
    #
   
