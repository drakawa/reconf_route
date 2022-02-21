c NPROCS = 1024 CLASS = A
c  
c  
c  This file is generated automatically by the setparams utility.
c  It sets the number of processors and the class of the NPB
c  in this directory. Do not modify it by hand.
c  
        integer            na, nonzer, niter
        double precision   shift, rcond
        parameter(  na=14000,
     >              nonzer=11,
     >              niter=15,
     >              shift=20.,
     >              rcond=1.0d-1 )

c number of nodes for which this version is compiled
        integer    nnodes_compiled
        parameter( nnodes_compiled = 1024)
        integer    num_proc_cols, num_proc_rows
        parameter( num_proc_cols=32, num_proc_rows=32 )
        logical  convertdouble
        parameter (convertdouble = .false.)
        character*11 compiletime
        parameter (compiletime='10 Nov 2020')
        character*5 npbversion
        parameter (npbversion='3.3.1')
        character*26 cs1
        parameter (cs1='${SIMGRID_PATH}/bin/smpiff')
        character*9 cs2
        parameter (cs2='$(MPIF77)')
        character*35 cs3
        parameter (cs3='-lsimgrid -lgfortran #-lsmpi -lgras')
        character*31 cs4
        parameter (cs4='-I${SIMGRID_PATH}/include/smpi/')
        character*2 cs5
        parameter (cs5='-O')
        character*24 cs6
        parameter (cs6='-O -L${SIMGRID_PATH}/lib')
        character*6 cs7
        parameter (cs7='randi8')
