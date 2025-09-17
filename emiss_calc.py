'''

This  program was originally part of the CHIANTI-VAMDC Interactive Python,
later renamed CHIANTI-IP, and part of CHIANTI Virtual IDL and Python (CHIANTI-VIP) 
software, an extension of the original CHIANTI software.


The original version was written in February 2016
within the UK-VAMDC consortium, with partial  funding (only for three months) by STFC (UK)
 by  Giulio Del Zanna (GDZ, University of Cambridge)

NAME: calc_emiss

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
         added photo-excitation and modified some functions.
         2,   4-Aug-2024, GDZ 

'''

import os
import ch_io # importing the CHIANTI-IP I/O and utilities
import ch_em # importing the CHIANTI-IP emissivity programs


def calc_emiss(ion_string, temperature=0, density=0,radTemperature=0,  rStar=0, proton_rates=0):

        # run the emissivity calculation 
    em=ch_em.emiss(ion_string,  eTemperature=temperature, 
                       eDensity=density, radTemperature= radTemperature, 
                       rStar=rStar, proton_rates=proton_rates)


    return em

    
