C FILE: SIM_EVO.F
C       If we ever need Poisson RVs, use this
C       https://hpaulkeeler.com/simulating-poisson-random-variables-in-fortran/
C       Also apparently modern fortran uses lowercase but we're sticking with all caps

C       Generate random integers in [A,B] from uniform distribution
        SUBROUTINE RANDINT(A, B, OUT)
        INTEGER, INTENT(IN) :: A, B
        INTEGER, INTENT(OUT) :: OUT
        REAL RAND_UNI
            CALL RANDOM_NUMBER(RAND_UNI)
            OUT = A + FLOOR((B + 1 - A) * RAND_UNI)
        END

C       Generate random nucleotide sequence
        SUBROUTINE GEN_SEQ(LEN, SEQ)
        IMPLICIT NONE
        INTEGER I
        INTEGER, INTENT(IN) :: LEN
        CHARACTER(LEN), INTENT(OUT) :: SEQ
        REAL RANDNS(LEN)
            CALL RANDOM_NUMBER(RANDNS)
            DO I=1,LEN
                IF (RANDNS(I) <= 0.25) THEN
                    SEQ(I:I) = 'A'
                ELSEIF (RANDNS(I) <= 0.5) THEN
                    SEQ(I:I) = 'C'
                ELSEIF (RANDNS(I) <= 0.75) THEN
                    SEQ(I:I) = 'G'
                ELSE
                    SEQ(I:I) = 'T'
                ENDIF
            ENDDO
        END

C       Full evolution simulation routine, adapted from gensynth.py
C       Output string, as far as I can tell, must be fixed length
C       I = insertion, D = deletion
        SUBROUTINE SIM_EVO(SIN, LOUT, PSUB, IDCNT, ICNT, IDSZS, SOUT)
        IMPLICIT NONE

        REAL, INTENT(IN) :: PSUB
        INTEGER, INTENT(IN) :: LOUT, IDCNT, ICNT, IDSZS(:)
        CHARACTER(*), INTENT(IN) :: SIN
        CHARACTER*(LOUT), INTENT(OUT) :: SOUT

        CHARACTER(:), ALLOCATABLE :: STMP, INSTMP
        REAL, ALLOCATABLE :: RAND_UNI_SUBS(:)
        INTEGER I, IDSZ, POS, L
C       Deletions
            STMP = TRIM(SIN)
            DO I=ICNT+1, IDCNT
                IDSZ = IDSZS(I)
                L = LEN(STMP) - IDSZ
                IF (L <= 0) THEN
                    STMP = ""
                ELSE
                    CALL RANDINT(1, L, POS)
                    STMP = STMP(:POS-1) // STMP(POS+IDSZ:)
                ENDIF
            ENDDO
C       Substitutions
            L = LEN(STMP)
            IF (L > 0) THEN
                ALLOCATE(RAND_UNI_SUBS(L))
                CALL RANDOM_NUMBER(RAND_UNI_SUBS)
                ALLOCATE(CHARACTER*(1) :: INSTMP)
                DO I=1,L
                    IF (RAND_UNI_SUBS(I) < PSUB) THEN
                        CALL GEN_SEQ(1, INSTMP)
                        STMP = STMP(:I-1) // INSTMP // STMP(I+1:)
                    ENDIF
                ENDDO
                DEALLOCATE(RAND_UNI_SUBS)
                DEALLOCATE(INSTMP)
            ENDIF
C       Insertions
            DO I=1,ICNT
                IDSZ = IDSZS(I)
                ALLOCATE(CHARACTER*(IDSZ) :: INSTMP)
                CALL GEN_SEQ(IDSZ, INSTMP)
                L = LEN(STMP)
                IF (L == 0) THEN
                    STMP = TRIM(INSTMP)
                ELSE
                    CALL RANDINT(1, L, POS)
                    STMP = STMP(:POS-1) // TRIM(INSTMP) // STMP(POS:)
                ENDIF
                DEALLOCATE(INSTMP)
            ENDDO
C       Pad if necessary (enforce_length)
            IF (LEN(STMP) < LOUT) THEN
                L = LOUT - LEN(STMP)
                ALLOCATE(CHARACTER*(L) :: INSTMP)
                CALL GEN_SEQ(L, INSTMP)
                CALL RANDINT(1, L, POS)
                STMP = INSTMP(:POS-1) // STMP // INSTMP(POS:)
            ENDIF
C       Truncate and "return"
            SOUT = STMP
            DEALLOCATE(STMP)
            
        END

C       Break into kmers, computing integer token IDs
        SUBROUTINE SEQ2KMERIDS(SEQ, L, K, TOKS)
        IMPLICIT NONE
        CHARACTER(*), INTENT(IN) :: SEQ
        INTEGER, INTENT(IN) :: L, K
        INTEGER, INTENT(OUT) :: TOKS(L - K + 1)
        INTEGER I, J, CVAL, KMERVAL
        CHARACTER C
            DO I=1,L - K + 1
                KMERVAL = 0
                DO J=0,K-1
                    C = SEQ(I+J:I+J)
                    IF (C == 'A') THEN
                        CVAL = 0
                    ELSEIF (C == 'C') THEN
                        CVAL = 1
                    ELSEIF (C == 'G') THEN
                        CVAL = 2
                    ELSE
                        CVAL = 3
                    ENDIF
                    KMERVAL = KMERVAL + CVAL * (K ** J)
                ENDDO
                TOKS(I) = KMERVAL + 1
            ENDDO
        END
C END FILE SIM_EVO.F