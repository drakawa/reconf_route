c NPROCS = 1024 CLASS = B
c  
c  
c  This file is generated automatically by the setparams utility.
c  It sets the number of processors and the class of the NPB
c  in this directory. Do not modify it by hand.
c  

c number of nodes for which this version is compiled
        integer nnodes_compiled, nnodes_xdim
        parameter (nnodes_compiled=1024, nnodes_xdim=32)

c full problem size
        integer isiz01, isiz02, isiz03
        parameter (isiz01=102, isiz02=102, isiz03=102)

c sub-domain array size
        integer isiz1, isiz2, isiz3
        parameter (isiz1=4, isiz2=4, isiz3=isiz03)

c number of iterations and how often to print the norm
        integer itmax_default, inorm_default
        parameter (itmax_default=250, inorm_default=250)
        double precision dt_default
        parameter (dt_default = 2.0d0)
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
