
C     PY ((21, 21), (-6, 6, 25)) : (21, 21, 25, 6, -6) # M0_ 1
C     PY ((2, -2), (-6, 6, 25)) : (2, -2, 25, 6, -6) # M1_ 1
C     PY ((4, -4), (-6, 6, 25)) : (4, -4, 25, 6, -6) # M1_ 1
C     PY ((1, -1), (-6, 6, 25)) : (1, -1, 25, 6, -6) # M1_ 1
C     PY ((3, -3), (-6, 6, 25)) : (3, -3, 25, 6, -6) # M1_ 1
      SUBROUTINE SMATRIXHEL(PDGS, PROCID, NPDG, P, ALPHAS, SCALE2,
     $  NHEL, ANS)
      IMPLICIT NONE
C     ALPHAS is given at scale2 (SHOULD be different of 0 for loop
C      induced, ignore for LO)  

CF2PY double precision, intent(in), dimension(0:3,npdg) :: p
CF2PY integer, intent(in), dimension(npdg) :: pdgs
CF2PY integer, intent(in):: procid
CF2PY integer, intent(in) :: npdg
CF2PY double precision, intent(out) :: ANS
CF2PY double precision, intent(in) :: ALPHAS
CF2PY double precision, intent(in) :: SCALE2
      INTEGER PDGS(*)
      INTEGER NPDG, NHEL, PROCID
      DOUBLE PRECISION P(*)
      DOUBLE PRECISION ANS, ALPHAS, PI,SCALE2
      INCLUDE 'coupl.inc'


      IF (SCALE2.EQ.0)THEN
        PI = 3.141592653589793D0
        G = 2* DSQRT(ALPHAS*PI)
        CALL UPDATE_AS_PARAM()
      ELSE
        CALL UPDATE_AS_PARAM2(SCALE2, ALPHAS)
      ENDIF

      IF(21.EQ.PDGS(1).AND.21.EQ.PDGS(2).AND.25.EQ.PDGS(3)
     $ .AND.6.EQ.PDGS(4).AND.-6.EQ.PDGS(5)
     $ .AND.(PROCID.LE.0.OR.PROCID.EQ.1)) THEN  ! 0
        CALL M0_SMATRIXHEL(P, NHEL, ANS)
      ELSE IF(2.EQ.PDGS(1).AND.-2.EQ.PDGS(2).AND.25.EQ.PDGS(3)
     $ .AND.6.EQ.PDGS(4).AND.-6.EQ.PDGS(5)
     $ .AND.(PROCID.LE.0.OR.PROCID.EQ.1)) THEN  ! 1
        CALL M1_SMATRIXHEL(P, NHEL, ANS)
      ELSE IF(4.EQ.PDGS(1).AND.-4.EQ.PDGS(2).AND.25.EQ.PDGS(3)
     $ .AND.6.EQ.PDGS(4).AND.-6.EQ.PDGS(5)
     $ .AND.(PROCID.LE.0.OR.PROCID.EQ.1)) THEN  ! 2
        CALL M1_SMATRIXHEL(P, NHEL, ANS)
      ELSE IF(1.EQ.PDGS(1).AND.-1.EQ.PDGS(2).AND.25.EQ.PDGS(3)
     $ .AND.6.EQ.PDGS(4).AND.-6.EQ.PDGS(5)
     $ .AND.(PROCID.LE.0.OR.PROCID.EQ.1)) THEN  ! 3
        CALL M1_SMATRIXHEL(P, NHEL, ANS)
      ELSE IF(3.EQ.PDGS(1).AND.-3.EQ.PDGS(2).AND.25.EQ.PDGS(3)
     $ .AND.6.EQ.PDGS(4).AND.-6.EQ.PDGS(5)
     $ .AND.(PROCID.LE.0.OR.PROCID.EQ.1)) THEN  ! 4
        CALL M1_SMATRIXHEL(P, NHEL, ANS)
      ENDIF

      RETURN
      END

      SUBROUTINE INITIALISE(PATH)
C     ROUTINE FOR F2PY to read the benchmark point.
      IMPLICIT NONE
      CHARACTER*512 PATH
CF2PY INTENT(IN) :: PATH
      CALL SETPARA(PATH)  !first call to setup the paramaters
      RETURN
      END


      SUBROUTINE CHANGE_PARA(NAME, VALUE)
      IMPLICIT NONE
CF2PY intent(in) :: name
CF2PY intent(in) :: value

      CHARACTER*512 NAME
      DOUBLE PRECISION VALUE

      LOGICAL M1_HELRESET
      COMMON /M1_HELRESET/ M1_HELRESET
      LOGICAL M0_HELRESET
      COMMON /M0_HELRESET/ M0_HELRESET

      INCLUDE '../Source/MODEL/input.inc'
      INCLUDE '../Source/MODEL/coupl.inc'

      M1_HELRESET = .TRUE.
      M0_HELRESET = .TRUE.

      SELECT CASE (NAME)
      CASE ('MB')
      MDL_MB = VALUE
      CASE ('MASS_5')
      MDL_MB = VALUE
      CASE ('MT')
      MDL_MT = VALUE
      CASE ('MASS_6')
      MDL_MT = VALUE
      CASE ('MTA')
      MDL_MTA = VALUE
      CASE ('MASS_15')
      MDL_MTA = VALUE
      CASE ('MZ')
      MDL_MZ = VALUE
      CASE ('MASS_23')
      MDL_MZ = VALUE
      CASE ('MH')
      MDL_MH = VALUE
      CASE ('MASS_25')
      MDL_MH = VALUE
      CASE ('aEWM1')
      AEWM1 = VALUE
      CASE ('SMINPUTS_1')
      AEWM1 = VALUE
      CASE ('Gf')
      MDL_GF = VALUE
      CASE ('SMINPUTS_2')
      MDL_GF = VALUE
      CASE ('aS')
      AS = VALUE
      CASE ('SMINPUTS_3')
      AS = VALUE
      CASE ('ymb')
      MDL_YMB = VALUE
      CASE ('YUKAWA_5')
      MDL_YMB = VALUE
      CASE ('ymt')
      MDL_YMT = VALUE
      CASE ('YUKAWA_6')
      MDL_YMT = VALUE
      CASE ('ymtau')
      MDL_YMTAU = VALUE
      CASE ('YUKAWA_15')
      MDL_YMTAU = VALUE
      CASE ('WT')
      MDL_WT = VALUE
      CASE ('DECAY_6')
      MDL_WT = VALUE
      CASE ('WZ')
      MDL_WZ = VALUE
      CASE ('DECAY_23')
      MDL_WZ = VALUE
      CASE ('WW')
      MDL_WW = VALUE
      CASE ('DECAY_24')
      MDL_WW = VALUE
      CASE ('WH')
      MDL_WH = VALUE
      CASE ('DECAY_25')
      MDL_WH = VALUE
      CASE DEFAULT
      WRITE(*,*) 'no parameter matching', NAME, VALUE
      END SELECT

      RETURN
      END

      SUBROUTINE UPDATE_ALL_COUP()
      IMPLICIT NONE
      CALL COUP()
      RETURN
      END


      SUBROUTINE GET_PDG_ORDER(PDG, ALLPROC)
      IMPLICIT NONE
CF2PY INTEGER, intent(out) :: PDG(5,5)
CF2PY INTEGER, intent(out) :: ALLPROC(5)
      INTEGER PDG(5,5), PDGS(5,5)
      INTEGER ALLPROC(5),PIDS(5)
      DATA PDGS/ 21,2,4,1,3,21,-2,-4,-1,-3,25,25,25,25,25,6,6,6,6,6,-6
     $ ,-6,-6,-6,-6 /
      DATA PIDS/ 1,1,1,1,1 /
      PDG = PDGS
      ALLPROC = PIDS
      RETURN
      END

      SUBROUTINE GET_PREFIX(PREFIX)
      IMPLICIT NONE
CF2PY CHARACTER*20, intent(out) :: PREFIX(5)
      CHARACTER*20 PREFIX(5),PREF(5)
      DATA PREF / 'M0_','M1_','M1_','M1_','M1_'/
      PREFIX = PREF
      RETURN
      END



      SUBROUTINE SET_FIXED_EXTRA_SCALE(NEW_VALUE)
      IMPLICIT NONE
CF2PY logical, intent(in) :: new_value
      LOGICAL NEW_VALUE
      LOGICAL FIXED_EXTRA_SCALE
      INTEGER MAXJETFLAVOR
      DOUBLE PRECISION MUE_OVER_REF
      DOUBLE PRECISION MUE_REF_FIXED
      COMMON/MODEL_SETUP_RUNNING/MAXJETFLAVOR,FIXED_EXTRA_SCALE
     $ ,MUE_OVER_REF,MUE_REF_FIXED

      FIXED_EXTRA_SCALE = NEW_VALUE
      RETURN
      END

      SUBROUTINE SET_MUE_OVER_REF(NEW_VALUE)
      IMPLICIT NONE
CF2PY double precision, intent(in) :: new_value
      DOUBLE PRECISION NEW_VALUE
      LOGICAL FIXED_EXTRA_SCALE
      INTEGER MAXJETFLAVOR
      DOUBLE PRECISION MUE_OVER_REF
      DOUBLE PRECISION MUE_REF_FIXED
      COMMON/MODEL_SETUP_RUNNING/MAXJETFLAVOR,FIXED_EXTRA_SCALE
     $ ,MUE_OVER_REF,MUE_REF_FIXED

      MUE_OVER_REF = NEW_VALUE

      RETURN
      END

      SUBROUTINE SET_MUE_REF_FIXED(NEW_VALUE)
      IMPLICIT NONE
CF2PY double precision, intent(in) :: new_value
      DOUBLE PRECISION NEW_VALUE
      LOGICAL FIXED_EXTRA_SCALE
      INTEGER MAXJETFLAVOR
      DOUBLE PRECISION MUE_OVER_REF
      DOUBLE PRECISION MUE_REF_FIXED
      COMMON/MODEL_SETUP_RUNNING/MAXJETFLAVOR,FIXED_EXTRA_SCALE
     $ ,MUE_OVER_REF,MUE_REF_FIXED

      MUE_REF_FIXED = NEW_VALUE

      RETURN
      END


      SUBROUTINE SET_MAXJETFLAVOR(NEW_VALUE)
      IMPLICIT NONE
CF2PY integer, intent(in) :: new_value
      INTEGER NEW_VALUE
      LOGICAL FIXED_EXTRA_SCALE
      INTEGER MAXJETFLAVOR
      DOUBLE PRECISION MUE_OVER_REF
      DOUBLE PRECISION MUE_REF_FIXED
      COMMON/MODEL_SETUP_RUNNING/MAXJETFLAVOR,FIXED_EXTRA_SCALE
     $ ,MUE_OVER_REF,MUE_REF_FIXED

      MAXJETFLAVOR = NEW_VALUE

      RETURN
      END


      SUBROUTINE SET_ASMZ(NEW_VALUE)
      IMPLICIT NONE
CF2PY double precision, intent(in) :: new_value
      DOUBLE PRECISION NEW_VALUE
      INTEGER NLOOP
      DOUBLE PRECISION ASMZ
      COMMON/A_BLOCK/ASMZ,NLOOP
      ASMZ = NEW_VALUE
      WRITE(*,*) 'asmz is set to ', NEW_VALUE

      RETURN
      END

      SUBROUTINE SET_NLOOP(NEW_VALUE)
      IMPLICIT NONE
CF2PY integer, intent(in) :: new_value
      INTEGER NEW_VALUE
      INTEGER NLOOP
      DOUBLE PRECISION ASMZ
      COMMON/A_BLOCK/ASMZ,NLOOP
      NLOOP = NEW_VALUE
      WRITE(*,*) 'nloop is set to ', NEW_VALUE

      RETURN
      END


