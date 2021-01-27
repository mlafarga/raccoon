!  f2py -m ccflibfort77 -c ccflibfort77.f
!----------------------------------------------------------------------

! Compute CCF
C Work in pixel space
C Mask width = 1 pixel
C Blaze weight fixed at the RV of the CCF center
      subroutine computeccf(w,f,b,ns,wm,fm,nm,rv,ccf,nrv)
      implicit none
      ! - Input
      integer,intent(in) :: ns !Spectrum points
      double precision,dimension(ns),intent(in) :: w,f !Spectrum
CC      double precision,dimension(ns),intent(in) :: wraw,f !Spectrum
      double precision,dimension(ns),intent(in) :: b !Blaze function
      integer,intent(in) :: nm !Mask points
      double precision,dimension(nm),intent(in) :: wm,fm !Mask
      integer,intent(in) :: nrv !RV points
      double precision,dimension(nrv),intent(in) :: rv !RV (km/s)
CC      double precision :: berv !BERV of the spectrum
      ! - Output
      double precision,dimension(nrv),intent(out) :: ccf
      ! - Local variables
      integer :: i,j,k,jin1
      ! Spectrum
      double precision,dimension(ns) :: pix !Spectrum pixels
CC      double precision,dimension(ns) :: w
      double precision,dimension(ns) :: w1,w2 !Pixel limits, in w space
      double precision :: stepj,stepmax !Pixel separation, in w space
      ! Blaze weight
      double precision :: rvmid !RV at the CCF center
      double precision,dimension(nm) :: wmmid !Mask lines shifted to rvmid
      double precision,dimension(nm) :: bw !Blaze weight corresponding to 
        !each shifted mask line
      double precision,dimension(nm) :: dbw !Blaze weight error
      ! Mask wavelength to pixel
      double precision :: yp1,ypn !Boundary conditions for spline subroutine
      double precision,dimension(ns) :: y2 !Second derivatives of 
        !interpolating function returned by spline subroutine and 
        !used when interpolating with splint subroutine
      ! Mask shift
      double precision :: wmsh !Mask lines shifted, in w space
      double precision :: pixmsh !Shifted mask line center, in pixel space
      double precision :: pixmsh1,pixmsh2 !Shifted mask line edges, in pixel space
      ! CCF
      double precision :: pix1,pix2
      double precision :: finterp !Flux center overlap region
      double precision :: dfinterp !Flux center overlap region error

Cf2py integer,intent(in,hide) :: ns
Cf2py integer,intent(in,hide) :: nm
Cf2py integer,intent(in,hide) :: nrv

      ! Pixel limits, in wavelength space
      do j=1,ns !for each pixel
        if(j.eq.1) then
         w2(j)=(w(j)+w(j+1))*0.5d0
         w1(j)=w(j)-(w(j+1)-w(j))*0.5d0
        elseif(j.eq.ns) then
         w1(j)=(w(j)+w(j-1))*0.5d0
         w2(j)=w(j)+(w(j)-w(j-1))*0.5d0
        else
         w1(j)=(w(j)+w(j-1))*0.5d0
         w2(j)=(w(j)+w(j+1))*0.5d0
        endif
      enddo
      ! Step max: maximum separation between consecutive pixels, 
      !in wavelength space
      !Used afterward to compute accelerator
      stepmax=0.d0
      do j=1,ns !for each pixel
       stepj=w2(j)-w1(j)
       if(stepj.gt.stepmax) then
        stepmax=stepj
       endif
      enddo

      ! Spectrum pixels
      do j=1,ns
       pix(j)=j-1
      enddo

      ! Blaze weight
      !Used when computing the CCF to take into account different SNR across
      !the spectrum
      !RV at the CCF center
      rvmid=rv(nrv/2.d0)
      do i=1,nm !for each mask line
       !Shift mask lines rvmid
       wmmid(i)=wm(i)*(1.d0+rvmid/2.99792458d5)
       !Find blaze value at shifted wavelength (bw)
       call interplin(w,b,ns,wmmid(i),bw(i),dbw(i))
      enddo

      ! Spline to transform from wavelength to pixel
      !Used afterwards to transform mask wavelengths to pixels
      yp1=1e31
      ypn=1e31
      call spline(w,pix,ns,yp1,ypn,y2)

      ! Compute CCF
      do k=1,nrv !for each RV

       ccf(k)=0.d0 !Initialize CCF to 0

       do i=1,nm !for each mask line

        ! Shift mask rv(k)
        wmsh=wm(i)*(1.d0+rv(k)/2.99792458d5)

        ! Transform mask wavelength to pixel
        call splint(w,pix,y2,ns,wmsh,pixmsh)

        ! Mask edges in pixel space: +/-0.5 pixel
        pixmsh1=pixmsh-0.5d0
        pixmsh2=pixmsh+0.5d0

        ! Check that mask line is inside spectrum limits
        if((pixmsh2).lt.(pix(1)-0.5d0)) goto 4
        if((pixmsh1).gt.(pix(ns)+0.5d0)) goto 4

        ! Accelerator
        !Skip pixels with w(j)<wmsh
        jin1=int((wmsh-stepmax-w(1))/stepmax)
        If(jin1.lt.1) jin1=1

        ! Compute CCF
        !Pixel space (spectrum and mask in pixels)
        !Interpolate flux at the middle of the overlap region between
        !the spectrum pixel and the mask line
        do j=jin1,ns !for each spectrum point
         !Spectrum pixel edges
         pix1=pix(j)-0.5d0
         pix2=pix(j)+0.5d0
         !No overlap between mask line i and spectrum pixel j
         !goto 3 => Skip rest of the pixels, because since pix is sorted,
         ! the remaining pixels will all fulfil pix1.gt.pixmsh2
         if(pix1.gt.pixmsh2) goto 3
         !Part of pixel inside mask
         if((pix2-pixmsh1)*(pix1-pixmsh2).lt.0.d0) then
          !Left end of pixel outside mask
          if((pix2-pixmsh1)*(pix1-pixmsh1).lt.0.d0)then
           !Left end of pixel outside mask, right end of pixel inside mask
           if(pix2.lt.pixmsh2) then
            call interplin(pix,f,ns,(pix2+pixmsh1)*0.5d0,finterp,
     &        dfinterp)
            ccf(k)=ccf(k)+finterp*fm(i)*(pix2-pixmsh1)*bw(i)
           !All mask inside pixel
           else
            call interplin(pix,f,ns,(pixmsh2+pixmsh1)*0.5d0,finterp,
     &        dfinterp)
            ccf(k)=ccf(k)+finterp*fm(i)*(pixmsh2-pixmsh1)*bw(i)
           endif
          !All pixel inside mask
          elseif((pix1-pixmsh1)*(pix2-pixmsh2).lt.0d0) then
           call interplin(pix,f,ns,(pix1+pix2)*0.5d0,finterp,dfinterp)
           ccf(k)=ccf(k)+finterp*fm(i)*bw(i)
          !Left end of pixel inside mask, right end of pixel outside mask
          else
           call interplin(pix,f,ns,(pixmsh2+pix1)*0.5d0,finterp,
     &      dfinterp)
           ccf(k)=ccf(k)+finterp*fm(i)*(pixmsh2-pix1)*bw(i)
          endif
         endif

        enddo !end spectrum points loop (j=jin1,ns-1)
 3      continue

 4     continue
       enddo !end mask lines loop (i=1,nm)


      enddo !end RV loop (k=1,nrv)
      return
      end
!----------------------------------------------------------------------


      subroutine computeccferr(w,f,b,ns,wm,fm,nm,rv,ccf,ccferr,nrv,ron)
      implicit none
      ! - Input
      integer,intent(in) :: ns !Spectrum points
      double precision,dimension(ns),intent(in) :: w,f !Spectrum
CC      double precision,dimension(ns),intent(in) :: wraw,f !Spectrum
      double precision,dimension(ns),intent(in) :: b !Blaze function
      integer,intent(in) :: nm !Mask points
      double precision,dimension(nm),intent(in) :: wm,fm !Mask
      integer,intent(in) :: nrv !RV points
      double precision,dimension(nrv),intent(in) :: rv !RV (km/s)
CC      double precision :: berv !BERV of the spectrum
      double precision,intent(in) :: ron !Readout noise
      ! - Output
      double precision,dimension(nrv),intent(out) :: ccf
      double precision,dimension(nrv),intent(out) :: ccferr
      ! - Local variables
      integer :: i,j,k,jin1
      ! Spectrum
      double precision,dimension(ns) :: pix !Spectrum pixels
CC      double precision,dimension(ns) :: w
      double precision,dimension(ns) :: w1,w2 !Pixel limits, in w space
      double precision :: stepj,stepmax !Pixel separation, in w space
      ! Blaze weight
      double precision :: rvmid !RV at the CCF center
      double precision,dimension(nm) :: wmmid !Mask lines shifted to rvmid
      double precision,dimension(nm) :: bw !Blaze weight corresponding to 
        !each shifted mask line
      double precision,dimension(nm) :: dbw !Blaze weight error 
      ! Mask wavelength to pixel
      double precision :: yp1,ypn !Boundary conditions for spline subroutine
      double precision,dimension(ns) :: y2 !Second derivatives of 
        !interpolating function returned by spline subroutine and 
        !used when interpolating with splint subroutine
      ! Mask shift
      double precision :: wmsh !Mask lines shifted, in w space
      double precision :: pixmsh !Shifted mask line center, in pixel space
      double precision :: pixmsh1,pixmsh2 !Shifted mask line edges, in pixel space
      ! CCF
      double precision :: pix1,pix2
      double precision :: finterp !Flux center overlap region
      double precision :: dfinterp !Flux center overlap region error

      ! RV error
      double precision,dimension(nrv) :: ccferr2 !ccferr**2
CCCC      double precision,dimension() :: ferr

Cf2py integer,intent(in,hide) :: ns
Cf2py integer,intent(in,hide) :: nm
Cf2py integer,intent(in,hide) :: nrv

      ! Pixel limits, in wavelength space
      do j=1,ns !for each pixel
        if(j.eq.1) then
         w2(j)=(w(j)+w(j+1))*0.5d0
         w1(j)=w(j)-(w(j+1)-w(j))*0.5d0
        elseif(j.eq.ns) then
         w1(j)=(w(j)+w(j-1))*0.5d0
         w2(j)=w(j)+(w(j)-w(j-1))*0.5d0
        else
         w1(j)=(w(j)+w(j-1))*0.5d0
         w2(j)=(w(j)+w(j+1))*0.5d0
        endif
      enddo
      ! Step max: maximum separation between consecutive pixels, 
      !in wavelength space
      !Used afterward to compute accelerator
      stepmax=0.d0
      do j=1,ns !for each pixel
       stepj=w2(j)-w1(j)
       if(stepj.gt.stepmax) then
        stepmax=stepj
       endif
      enddo

      ! Spectrum pixels
      do j=1,ns
       pix(j)=j-1
      enddo

      ! Blaze weight
      !Used when computing the CCF to take into account different SNR across
      !the spectrum
      !RV at the CCF center
      rvmid=rv(nrv/2.d0)
      do i=1,nm !for each mask line
       !Shift mask lines rvmid
       wmmid(i)=wm(i)*(1.d0+rvmid/2.99792458d5)
       !Find blaze value at shifted wavelength (bw)
       call interplin(w,b,ns,wmmid(i),bw(i),dbw(i))
      enddo

      ! Spline to transform from wavelength to pixel
      !Used afterwards to transform mask wavelengths to pixels
      yp1=1e31
      ypn=1e31
      call spline(w,pix,ns,yp1,ypn,y2)

      ! Compute CCF
      do k=1,nrv !for each RV

       ccf(k)=0.d0 !Initialize CCF to 0
       ccferr2(k)=0.d0

       do i=1,nm !for each mask line

        ! Shift mask rv(k)
        wmsh=wm(i)*(1.d0+rv(k)/2.99792458d5)

        ! Transform mask wavelength to pixel
        call splint(w,pix,y2,ns,wmsh,pixmsh)

        ! Mask edges in pixel space: +/-0.5 pixel
        pixmsh1=pixmsh-0.5d0
        pixmsh2=pixmsh+0.5d0

        ! Check that mask line is inside spectrum limits
C         if((pixmsh2).lt.(pix(0)-0.5d0)) goto 4
        if((pixmsh2).lt.(pix(1)-0.5d0)) goto 4
C         if((pixmsh1).gt.(pix(ns-1)+0.5d0)) goto 4
        if((pixmsh1).gt.(pix(ns)+0.5d0)) goto 4

        ! Accelerator
        !Skip pixels with w(j)<wmsh
        jin1=int((wmsh-stepmax-w(1))/stepmax)
        If(jin1.lt.1) jin1=1

        ! Compute CCF
        !Pixel space (spectrum and mask in pixels)
        !Interpolate flux at the middle of the overlap region between
        !the spectrum pixel and the mask line
C         do j=jin1,ns-1 !for each spectrum point
        do j=jin1,ns !for each spectrum point

         !Spectrum pixel edges
         pix1=pix(j)-0.5d0
         pix2=pix(j)+0.5d0
         !No overlap between mask line i and spectrum pixel j
         !goto 3 => Skip rest of the pixels, because since pix is sorted,
         ! the remaining pixels will all fulfil pix1.gt.pixmsh2
         if(pix1.gt.pixmsh2) goto 3
         !Part of pixel inside mask
         if((pix2-pixmsh1)*(pix1-pixmsh2).lt.0.d0) then
          !Left end of pixel outside mask
          if((pix2-pixmsh1)*(pix1-pixmsh1).lt.0.d0)then
           !Left end of pixel outside mask, right end of pixel inside mask
           if(pix2.lt.pixmsh2) then
            call interplin(pix,f,ns,(pix2+pixmsh1)*0.5d0,finterp,
     &        dfinterp)
            ccf(k)=ccf(k)+finterp*fm(i)*(pix2-pixmsh1)*bw(i)
            ccferr2(k)=ccferr2(k)+finterp*(pix2-pixmsh1)*bw(i)+ron**2
           !All mask inside pixel
           else
            call interplin(pix,f,ns,(pixmsh2+pixmsh1)*0.5d0,finterp,
     &        dfinterp)
            ccf(k)=ccf(k)+finterp*fm(i)*(pixmsh2-pixmsh1)*bw(i)
            ccferr2(k)=ccferr2(k)+finterp*(pixmsh2-pixmsh1)*bw(i)+ron**2
           endif
          !All pixel inside mask
          elseif((pix1-pixmsh1)*(pix2-pixmsh2).lt.0d0) then
           call interplin(pix,f,ns,(pix1+pix2)*0.5d0,finterp,dfinterp)
           ccf(k)=ccf(k)+finterp*fm(i)*bw(i)
           ccferr2(k)=ccferr2(k)+finterp*bw(i)+ron**2
          !Left end of pixel inside mask, right end of pixel outside mask
          else
           call interplin(pix,f,ns,(pixmsh2+pix1)*0.5d0,finterp,
     &       dfinterp)
           ccf(k)=ccf(k)+finterp*fm(i)*(pixmsh2-pix1)*bw(i)
           ccferr2(k)=ccferr2(k)+finterp*(pixmsh2-pix1)*bw(i)+ron**2
          endif
         endif

        enddo !end spectrum points loop (j=jin1,ns-1)
 3      continue

c       write(*,*)'  ',ccferr2(k)

 4     continue
       enddo !end mask lines loop (i=1,nm)


       ccferr(k)=dsqrt(ccferr2(k))

      enddo !end RV loop (k=1,nrv)
      return
      end
!----------------------------------------------------------------------

      subroutine computeccferrfluxorig(w,f,b,forig,ns,wm,fm,nm,
     &   rv,ccf,ccferr,nrv,ron)
      implicit none
      ! - Input
      integer,intent(in) :: ns !Spectrum points
      double precision,dimension(ns),intent(in) :: w,f,forig !Spectrum
CC      double precision,dimension(ns),intent(in) :: wraw,f !Spectrum
      double precision,dimension(ns),intent(in) :: b !Blaze function
      integer,intent(in) :: nm !Mask points
      double precision,dimension(nm),intent(in) :: wm,fm !Mask
      integer,intent(in) :: nrv !RV points
      double precision,dimension(nrv),intent(in) :: rv !RV (km/s)
CC      double precision :: berv !BERV of the spectrum
      double precision,intent(in) :: ron !Readout noise
      ! - Output
      double precision,dimension(nrv),intent(out) :: ccf
      double precision,dimension(nrv),intent(out) :: ccferr
      ! - Local variables
      integer :: i,j,k,jin1
      ! Spectrum
      double precision,dimension(ns) :: pix !Spectrum pixels
CC      double precision,dimension(ns) :: w
      double precision,dimension(ns) :: w1,w2 !Pixel limits, in w space
      double precision :: stepj,stepmax !Pixel separation, in w space
      ! Blaze weight
      double precision :: rvmid !RV at the CCF center
      double precision,dimension(nm) :: wmmid !Mask lines shifted to rvmid
      double precision,dimension(nm) :: bw !Blaze weight corresponding to 
        !each shifted mask line
      double precision,dimension(nm) :: dbw !Blaze weight error 
      ! Mask wavelength to pixel
      double precision :: yp1,ypn !Boundary conditions for spline subroutine
      double precision,dimension(ns) :: y2 !Second derivatives of 
        !interpolating function returned by spline subroutine and 
        !used when interpolating with splint subroutine
      ! Mask shift
      double precision :: wmsh !Mask lines shifted, in w space
      double precision :: pixmsh !Shifted mask line center, in pixel space
      double precision :: pixmsh1,pixmsh2 !Shifted mask line edges, in pixel space
      ! CCF
      double precision :: pix1,pix2
      double precision :: finterp,foriginterp !Flux center overlap region
      double precision :: dfinterp,dforiginterp !Flux center overlap region error

      ! RV error
      double precision,dimension(nrv) :: ccferr2 !ccferr**2
CCCC      double precision,dimension() :: ferr

Cf2py integer,intent(in,hide) :: ns
Cf2py integer,intent(in,hide) :: nm
Cf2py integer,intent(in,hide) :: nrv

      ! Pixel limits, in wavelength space
      do j=1,ns !for each pixel
        if(j.eq.1) then
         w2(j)=(w(j)+w(j+1))*0.5d0
         w1(j)=w(j)-(w(j+1)-w(j))*0.5d0
        elseif(j.eq.ns) then
         w1(j)=(w(j)+w(j-1))*0.5d0
         w2(j)=w(j)+(w(j)-w(j-1))*0.5d0
        else
         w1(j)=(w(j)+w(j-1))*0.5d0
         w2(j)=(w(j)+w(j+1))*0.5d0
        endif
      enddo
      ! Step max: maximum separation between consecutive pixels, 
      !in wavelength space
      !Used afterward to compute accelerator
      stepmax=0.d0
      do j=1,ns !for each pixel
       stepj=w2(j)-w1(j)
       if(stepj.gt.stepmax) then
        stepmax=stepj
       endif
      enddo

      ! Spectrum pixels
      do j=1,ns
       pix(j)=j-1
      enddo

      ! Blaze weight
      !Used when computing the CCF to take into account different SNR across
      !the spectrum
      !RV at the CCF center
      rvmid=rv(nrv/2.d0)
      do i=1,nm !for each mask line
       !Shift mask lines rvmid
       wmmid(i)=wm(i)*(1.d0+rvmid/2.99792458d5)
       !Find blaze value at shifted wavelength (bw)
       call interplin(w,b,ns,wmmid(i),bw(i),dbw(i))
      enddo

      ! Spline to transform from wavelength to pixel
      !Used afterwards to transform mask wavelengths to pixels
      yp1=1e31
      ypn=1e31
      call spline(w,pix,ns,yp1,ypn,y2)

      ! Compute CCF
      do k=1,nrv !for each RV

       ccf(k)=0.d0 !Initialize CCF to 0
       ccferr2(k)=0.d0

       do i=1,nm !for each mask line

        ! Shift mask rv(k)
        wmsh=wm(i)*(1.d0+rv(k)/2.99792458d5)

        ! Transform mask wavelength to pixel
        call splint(w,pix,y2,ns,wmsh,pixmsh)

        ! Mask edges in pixel space: +/-0.5 pixel
        pixmsh1=pixmsh-0.5d0
        pixmsh2=pixmsh+0.5d0

        ! Check that mask line is inside spectrum limits
        if((pixmsh2).lt.(pix(1)-0.5d0)) goto 4
        if((pixmsh1).gt.(pix(ns)+0.5d0)) goto 4


        ! Accelerator
        !Skip pixels with w(j)<wmsh
        jin1=int((wmsh-stepmax-w(1))/stepmax)
        If(jin1.lt.1) jin1=1

        ! Compute CCF
        !Pixel space (spectrum and mask in pixels)
        !Interpolate flux at the middle of the overlap region between
        !the spectrum pixel and the mask line
        do j=jin1,ns !for each spectrum point
         !Spectrum pixel edges
         pix1=pix(j)-0.5d0
         pix2=pix(j)+0.5d0
         !No overlap between mask line i and spectrum pixel j
         !goto 3 => Skip rest of the pixels, because since pix is sorted,
         ! the remaining pixels will all fulfil pix1.gt.pixmsh2
         if(pix1.gt.pixmsh2) goto 3
         !Part of pixel inside mask
         if((pix2-pixmsh1)*(pix1-pixmsh2).lt.0.d0) then
          !Left end of pixel outside mask
          if((pix2-pixmsh1)*(pix1-pixmsh1).lt.0.d0)then
           !Left end of pixel outside mask, right end of pixel inside mask
           if(pix2.lt.pixmsh2) then
            call interplin(pix,f,ns,(pix2+pixmsh1)*0.5d0,finterp,
     &        dfinterp)
            ccf(k)=ccf(k)+finterp*fm(i)*(pix2-pixmsh1)*bw(i)
            call interplin(pix,forig,ns,(pix2+pixmsh1)*0.5d0,
     &        foriginterp,dforiginterp)
            ccferr2(k)=ccferr2(k)+foriginterp*(pix2-pixmsh1)*bw(i)
     &        +ron**2
           !All mask inside pixel
           else
            call interplin(pix,f,ns,(pixmsh2+pixmsh1)*0.5d0,finterp,
     &        dfinterp)
            ccf(k)=ccf(k)+finterp*fm(i)*(pixmsh2-pixmsh1)*bw(i)
            call interplin(pix,forig,ns,(pixmsh2+pixmsh1)*0.5d0,
     &        foriginterp,dforiginterp)
            ccferr2(k)=ccferr2(k)+foriginterp*(pixmsh2-pixmsh1)*bw(i)
     &        +ron**2
           endif
          !All pixel inside mask
          elseif((pix1-pixmsh1)*(pix2-pixmsh2).lt.0d0) then
           call interplin(pix,f,ns,(pix1+pix2)*0.5d0,finterp,dfinterp)
           ccf(k)=ccf(k)+finterp*fm(i)*bw(i)
           call interplin(pix,forig,ns,(pix1+pix2)*0.5d0,foriginterp,
     &      dforiginterp)
           ccferr2(k)=ccferr2(k)+foriginterp*bw(i)+ron**2
          !Left end of pixel inside mask, right end of pixel outside mask
          else
           call interplin(pix,f,ns,(pixmsh2+pix1)*0.5d0,finterp,
     &       dfinterp)
           ccf(k)=ccf(k)+finterp*fm(i)*(pixmsh2-pix1)*bw(i)
           call interplin(pix,forig,ns,(pixmsh2+pix1)*0.5d0,
     &       foriginterp,dforiginterp)
           ccferr2(k)=ccferr2(k)+foriginterp*(pixmsh2-pix1)*bw(i)
     &       +ron**2
          endif
         endif

        enddo !end spectrum points loop (j=jin1,ns-1)
 3      continue

c       write(*,*)'  ',ccferr2(k)

 4     continue
       enddo !end mask lines loop (i=1,nm)


       ccferr(k)=dsqrt(ccferr2(k))

      enddo !end RV loop (k=1,nrv)
      return
      end
!----------------------------------------------------------------------


! Fit CCF
! x=rv, y=ccf, nd=nrv
      subroutine fitccf(x,y,sig,nd,fitrng,ain,lista,np,a,da,funcfitnam)

      IMPLICIT REAL*8 (A-H,O-Z)

      ! Input
      integer,intent(in) :: nd
      double precision,dimension(nd),intent(in) :: x,y
      double precision,dimension(nd),intent(inout) :: sig
      double precision,intent(in) :: fitrng !Fit [fitcen-fitrng,fitcen+fitrng]
      character*50,intent(in) :: funcfitnam
      integer,intent(in) :: np
      double precision,dimension(np),intent(in) :: ain !Initial param vals
      integer,dimension(np),intent(in) :: lista
      ! Output
c      double precision,dimension(nd),intent(out) :: fit
      double precision,dimension(np),intent(out) :: a,da
c      double precision,intent(out) :: rvmax,contrast,fwhm
      ! Local
      double precision :: gausmx,rvmx !CCF minimum
      integer :: nco !Number of data points to fit
      double precision,dimension(nd) :: xf,yf,sigf !Input data inside fitting range

      integer :: i,j,k
      integer :: itmax!, np
      double precision :: alamda,chisq,oldchisq
c      integer,dimension(20) :: lista
      double precision,dimension(20) :: dyda
      double precision,dimension(20,20) :: covar,alpha
c      double precision :: pi

Cf2py integer,intent(in,hide) :: nd
Cf2py integer,intent(in,hide) :: np

      ! Find CCF min: (rvmx,gausmx)
      gausmx=+1.d30
      do i=1,nd !for each CCF point
       if(y(i).lt.gausmx) then
        gausmx=y(i)
        rvmx=x(i)
       endif
      enddo

      ! Select data points to fit (data in fitting range): xf, yf
      !nd: Total number of points of RV (len(rv))
      !nco: Number of points RV fit (len(rvmx-fitrng,rvmx+fitrng))
      !xf(nco), yf(nco)
      do i=1,nd !for each CCF point
       if (x(i).gt.rvmx-fitrng) then !if x(i) inside fit range
       nco=0
       do j=1,nd-i+1
        nco=nco+1
        xf(j)=x(i+j-1)
        yf(j)=y(i+j-1)
        sigf(j)=sig(i+j-1)
        if(x(i+j).gt.rvmx+fitrng) goto 10
c        if(x(i+j-1).gt.rvmx+fitrng) goto 10
       enddo
       goto 10
       endif
      enddo
 10   continue

c      ! Initialize sigma (arbitrary value)
c      do i=1,nco
c       sig(i)=0.01d0
c      enddo

      ! Initialize Gaussian parameters
c      LISTA(4)=1
c      A(4)=1.d0
      A(4)=ain(4)
c      LISTA(1)=1
c      A(1)=gausmx-A(4)
      A(1)=ain(1)
c      LISTA(2)=1
c      A(2)=rvmx
      A(2)=ain(2)
c      LISTA(3)=1
c      A(3)=5.d0
      A(3)=ain(3)

      ! Fit
c      NP=4
      ITMAX=10000000
      ALAMDA=-0.1D0

      OLDCHISQ=0.D0
      ICO=0
      Do 15 k=1,ITMAX
      CALL MRQMIN(XF,YF,SIGF,NCO,A,LISTA,NP,COVAR,ALPHA,20,
     $CHISQ,ALAMDA,funcfitnam)
      IF(DABS(CHISQ-OLDCHISQ).LT.1.D-6) THEN
        ICO=ICO+1
      ELSE
        ICO=0
      ENDIF
      IF(ICO.EQ.5) GOTO 16
      OLDCHISQ=CHISQ
c 68   Format('Reduced Chi square = ',f13.8)
 15   Continue
      Write(*,*) 'Max. number of iterations exceeded'

 16   continue

      do 90 i=1,nco
 90    sig(i)=sigf(i)*dsqrt(CHISQ/(nco-4))


      ! Recompute fit with correct sig
      CALL MRQMIN(XF,YF,sig,NCO,A,LISTA,NP,COVAR,ALPHA,20,
     $CHISQ,ALAMDA,funcfitnam)
c      write(*,*) 'sig_new',sig(1),'schi2_new',dsqrt(CHISQ/(nco-4))

      ! Call MRQMIN one final time with ALAMDA=0.0 so that array COVAR will
      ! return the covariance matrix
      CALL MRQMIN(XF,YF,SIGF,NCO,A,LISTA,NP,COVAR,ALPHA,20,
     $CHISQ,0.D0,funcfitnam)
c      CALL MRQMIN(XF,YF,SIG_cent,NCO,A,LISTA,NP,COVAR,ALPHA,20,
c     $CHISQ,0.D0)

      ! Gaussian parameters uncertainties
      !Derived from the square roots of the diagonal elements of COVAR
      da(1)=dsqrt(covar(1,1))
      da(2)=dsqrt(covar(2,2)) !sig RVmaximum
      da(3)=dsqrt(covar(3,3))
      da(4)=dsqrt(covar(4,4))
      
C      !Gaussian: Y=A(4)+A(1)*dexp(-(X-A(2))**2.d0/A(3))
C      contrast=100.d0*A(1)
C      rvmaximum=A(2)
Cc      FWHM=A(3)
C      fwhm=2*dsqrt(log(2)*A(3))
C      fitmax=A(4)
C      ! FWHM=2*dsqrt(log(2)*A(3))
C      !log: natural

C      write(*,*) A(1),A(2),A(3),A(4)

      return
      END
!----------------------------------------------------------------------


! Fit range absolute max-max
      subroutine fitccfmaxabs(x,y,sig,nd,ain,lista,np,a,da,funcfitnam)

      IMPLICIT REAL*8 (A-H,O-Z)

      ! Input
      integer,intent(in) :: nd
      double precision,dimension(nd),intent(in) :: x,y
      double precision,dimension(nd),intent(inout) :: sig
c      double precision,intent(in) :: fitrng !Fit [fitcen-fitrng,fitcen+fitrng]
      character*50,intent(in) :: funcfitnam
      integer,intent(in) :: np
      double precision,dimension(np),intent(in) :: ain !Initial param vals
      integer,dimension(np),intent(in) :: lista
      ! Output
c      double precision,dimension(nd),intent(out) :: fit
      double precision,dimension(np),intent(out) :: a,da
c      double precision,intent(out) :: rvmax,contrast,fwhm
      ! Local
      double precision :: gausmx,rvmx !CCF minimum
      integer :: nco !Number of data points to fit
      double precision,dimension(nd) :: xf,yf,sigf !Input data inside fitting range
      double precision :: ccfal,ccfalr,rvmaxl,rvmaxr !CCF max

      integer :: i,j,k
      integer :: itmax!, np
      double precision :: alamda,chisq,oldchisq
c      integer,dimension(20) :: lista
      double precision,dimension(20) :: dyda
      double precision,dimension(20,20) :: covar,alpha
c      double precision :: pi

Cf2py integer,intent(in,hide) :: nd
Cf2py integer,intent(in,hide) :: np

      ! Find CCF min: (rvmx,gausmx)
      gausmx=+1.d30
      do i=1,nd !for each CCF point
       if(y(i).lt.gausmx) then
        gausmx=y(i)
        rvmx=x(i)
       endif
      enddo

      ! Find CCF maxima
      ccfal=0.d0
      ccfar=0.d0
      rvmaxl=0.d0
      rvmaxr=0.d0
      do i=1,nd !for each CCF point
       if(y(i).gt.ccfal.and.x(i).lt.rvmx) then
        ccfal=y(i)
        rvmaxl=x(i)
       endif
       if(y(i).gt.ccfar.and.x(i).gt.rvmx) then
        ccfar=y(i)
        rvmaxr=x(i)
       endif
      enddo

      ! Select data points to fit (data in fitting range): xf, yf
      !nd: Total number of points of RV (len(rv))
      !nco: Number of points RV fit (len(rvmaxl,rvmaxr))
      !xf(nco), yf(nco)
      do i=1,nd !for each CCF point
       if (x(i).gt.rvmaxl) then !if x(i) inside fit range
       nco=0
       do j=1,nd-i+1
        nco=nco+1
        xf(j)=x(i+j-1)
        yf(j)=y(i+j-1)
        sigf(j)=sig(i+j-1)
        if(x(i+j).gt.rvmaxr) goto 10
c        if(x(i+j-1).gt.rvmx+fitrng) goto 10
       enddo
       goto 10
       endif
      enddo
 10   continue

c      ! Select data points to fit (data in fitting range): xf, yf
c      !nd: Total number of points of RV (len(rv))
c      !nco: Number of points RV fit (len(rvmx-fitrng,rvmx+fitrng))
c      !xf(nco), yf(nco)
c      do i=1,nd !for each CCF point
c       if (x(i).gt.rvmx-fitrng) then !if x(i) inside fit range
c       nco=0
c       do j=1,nd-i+1
c        nco=nco+1
c        xf(j)=x(i+j-1)
c        yf(j)=y(i+j-1)
c        sigf(j)=sig(i+j-1)
c        if(x(i+j).gt.rvmx+fitrng) goto 10
cc        if(x(i+j-1).gt.rvmx+fitrng) goto 10
c       enddo
c       goto 10
c       endif
c      enddo
c 10   continue

c      ! Initialize sigma (arbitrary value)
c      do i=1,nco
c       sig(i)=0.01d0
c      enddo

      ! Initialize Gaussian parameters
c      LISTA(4)=1
c      A(4)=1.d0
      A(4)=ain(4)
c      LISTA(1)=1
c      A(1)=gausmx-A(4)
      A(1)=ain(1)
c      LISTA(2)=1
c      A(2)=rvmx
      A(2)=ain(2)
c      LISTA(3)=1
c      A(3)=5.d0
      A(3)=ain(3)

      ! Fit
c      NP=4
      ITMAX=10000000
      ALAMDA=-0.1D0

      OLDCHISQ=0.D0
      ICO=0
      Do 15 k=1,ITMAX
      CALL MRQMIN(XF,YF,SIGF,NCO,A,LISTA,NP,COVAR,ALPHA,20,
     $CHISQ,ALAMDA,funcfitnam)
      IF(DABS(CHISQ-OLDCHISQ).LT.1.D-6) THEN
        ICO=ICO+1
      ELSE
        ICO=0
      ENDIF
      IF(ICO.EQ.5) GOTO 16
      OLDCHISQ=CHISQ
c 68   Format('Reduced Chi square = ',f13.8)
 15   Continue
      Write(*,*) 'Max. number of iterations exceeded'

 16   continue

      ! Modify sig so that red chi2 equals 1
      do 90 i=1,nco
 90    sigf(i)=sigf(i)*dsqrt(CHISQ/(nco-4))

c      write(*,*)'sig',sigf(1),'redchi2',dsqrt(CHISQ/(nco-4))
c      write(*,*)'rverr',dsqrt(covar(2,2)),'covar',covar(2,1)

c      chisq=1.d0*(nco-4)
c      write(*,*)'sig',sigf(1),'redchi2',dsqrt(CHISQ/(nco-4))

      ! Recompute fit with correct sig
      CALL MRQMIN(XF,YF,sigf,NCO,A,LISTA,NP,COVAR,ALPHA,20,
     $CHISQ,ALAMDA,funcfitnam)
      ! Only need to call once. Same as doing the whole loop again, 
      ! because we already have the best fit values in `a`.

c      write(*,*) 'sig_new',sigf(1),'redchi2_new',dsqrt(CHISQ/(nco-4))
c      write(*,*)'rverr',dsqrt(covar(2,2)),'covar',covar(2,1)

      ! Call MRQMIN one final time with ALAMDA=0.0 so that array COVAR will
      ! return the covariance matrix
C      CALL MRQMIN(XF,YF,SIGF,NCO,A,LISTA,NP,COVAR,ALPHA,20,
C     $CHISQ,0.D0,funcfitnam)
      CALL MRQMIN(XF,YF,SIGF,NCO,A,LISTA,NP,COVAR,ALPHA,20,
     $CHISQ,0.D0,funcfitnam)
c      CALL MRQMIN(XF,YF,SIG_cent,NCO,A,LISTA,NP,COVAR,ALPHA,20,
c     $CHISQ,0.D0)

c      write(*,*)'sig',sigf(1),'redchi2',dsqrt(CHISQ/(nco-4))
c      write(*,*)a(1),a(2),a(3),a(4)
c      write(*,*)'rverr',dsqrt(covar(2,2)),'covar',covar(2,1)

      ! Gaussian parameters uncertainties
      !Derived from the square roots of the diagonal elements of COVAR
      da(1)=dsqrt(covar(1,1))
      da(2)=dsqrt(covar(2,2)) !sig RVmaximum
      da(3)=dsqrt(covar(3,3))
      da(4)=dsqrt(covar(4,4))

C      !Gaussian: Y=A(4)+A(1)*dexp(-(X-A(2))**2.d0/A(3))
C      contrast=100.d0*A(1)
C      rvmaximum=A(2)
Cc      FWHM=A(3)
C      fwhm=2*dsqrt(log(2)*A(3))
C      fitmax=A(4)
C      ! FWHM=2*dsqrt(log(2)*A(3))
C      !log: natural

C      write(*,*) A(1),A(2),A(3),A(4)

      return
      END
!----------------------------------------------------------------------

! Fit range max-max closest to minimum
      subroutine fitccfmax(x,y,sig,nd,ain,lista,np,a,da,funcfitnam)

      IMPLICIT REAL*8 (A-H,O-Z)

      ! Input
      integer,intent(in) :: nd
      double precision,dimension(nd),intent(in) :: x,y
      double precision,dimension(nd),intent(inout) :: sig
c      double precision,intent(in) :: fitrng !Fit [fitcen-fitrng,fitcen+fitrng]
      character*50,intent(in) :: funcfitnam
      integer,intent(in) :: np
      double precision,dimension(np),intent(in) :: ain !Initial param vals
      integer,dimension(np),intent(in) :: lista
      ! Output
c      double precision,dimension(nd),intent(out) :: fit
      double precision,dimension(np),intent(out) :: a,da
c      double precision,intent(out) :: rvmax,contrast,fwhm
      ! Local
      double precision :: gausmx,rvmx !CCF minimum
c      integer :: idxmin !CCF minimum index in x,y arrays
      integer :: nco !Number of data points to fit
      double precision,dimension(nd) :: xf,yf,sigf !Input data inside fitting range
      double precision :: ccfal,ccfalr,rvmaxl,rvmaxr !CCF max

      integer :: i,j,k
      integer :: itmax!, np
      double precision :: alamda,chisq,oldchisq
c      integer,dimension(20) :: lista
      double precision,dimension(20) :: dyda
      double precision,dimension(20,20) :: covar,alpha
c      double precision :: pi

Cf2py integer,intent(in,hide) :: nd
Cf2py integer,intent(in,hide) :: np

      ! Find CCF min: (rvmx,gausmx)
      gausmx=+1.d30
      do i=1,nd !for each CCF point
       if(y(i).lt.gausmx) then
        gausmx=y(i)
        rvmx=x(i)
c        idxmin=i
       endif
      enddo

C      ! Find CCF absolute maxima
C      ccfal=0.d0
C      ccfar=0.d0
C      rvmaxl=0.d0
C      rvmaxr=0.d0
C      do i=1,nd !for each CCF point
C       if(y(i).gt.ccfal.and.x(i).lt.rvmx) then
C        ccfal=y(i)
C        rvmaxl=x(i)
C       endif
C       if(y(i).gt.ccfar.and.x(i).gt.rvmx) then
C        ccfar=y(i)
C        rvmaxr=x(i)
C       endif
C      enddo

      ! Find CCF maxima closest to CCF minimum
      ! - Right maximum
      ccfar=0.d0
      rvmaxr=0.d0
      do i=1,nd !for each CCF point
       if(y(i).gt.ccfar.and.x(i).gt.rvmx) then
        ccfar=y(i)
        rvmaxr=x(i)
        !Stop loop if next CCF point is smaller
        if(i.lt.nd) then
         if(y(i+1).lt.ccfar) then
          exit
         endif
        endif
       endif
      enddo
      ! - Left maximum
      ccfal=0.d0
      rvmaxl=0.d0
      do i=nd,1,-1 !for each CCF point
       if(y(i).gt.ccfal.and.x(i).lt.rvmx) then
        ccfal=y(i)
        rvmaxl=x(i)
        !Stop loop if next CCF point is smaller
        if(i.gt.1) then
         if(y(i-1).lt.ccfal) then
          exit
         endif
        endif
       endif
      enddo

      ! Select data points to fit (data in fitting range): xf, yf
      !nd: Total number of points of RV (len(rv))
      !nco: Number of points RV fit (len(rvmaxl,rvmaxr))
      !xf(nco), yf(nco)
      do i=1,nd !for each CCF point
       if (x(i).gt.rvmaxl) then !if x(i) inside fit range
       nco=0
       do j=1,nd-i+1
        nco=nco+1
        xf(j)=x(i+j-1)
        yf(j)=y(i+j-1)
        sigf(j)=sig(i+j-1)
        if(x(i+j).gt.rvmaxr) goto 10
c        if(x(i+j-1).gt.rvmx+fitrng) goto 10
       enddo
       goto 10
       endif
      enddo
 10   continue

c      ! Select data points to fit (data in fitting range): xf, yf
c      !nd: Total number of points of RV (len(rv))
c      !nco: Number of points RV fit (len(rvmx-fitrng,rvmx+fitrng))
c      !xf(nco), yf(nco)
c      do i=1,nd !for each CCF point
c       if (x(i).gt.rvmx-fitrng) then !if x(i) inside fit range
c       nco=0
c       do j=1,nd-i+1
c        nco=nco+1
c        xf(j)=x(i+j-1)
c        yf(j)=y(i+j-1)
c        sigf(j)=sig(i+j-1)
c        if(x(i+j).gt.rvmx+fitrng) goto 10
cc        if(x(i+j-1).gt.rvmx+fitrng) goto 10
c       enddo
c       goto 10
c       endif
c      enddo
c 10   continue

c      ! Initialize sigma (arbitrary value)
c      do i=1,nco
c       sig(i)=0.01d0
c      enddo

      ! Initialize Gaussian parameters
c      LISTA(4)=1
c      A(4)=1.d0
      A(4)=ain(4)
c      LISTA(1)=1
c      A(1)=gausmx-A(4)
      A(1)=ain(1)
c      LISTA(2)=1
c      A(2)=rvmx
      A(2)=ain(2)
c      LISTA(3)=1
c      A(3)=5.d0
      A(3)=ain(3)

      ! Fit
c      NP=4
      ITMAX=10000000
      ALAMDA=-0.1D0

      OLDCHISQ=0.D0
      ICO=0
      Do 15 k=1,ITMAX
      CALL MRQMIN(XF,YF,SIGF,NCO,A,LISTA,NP,COVAR,ALPHA,20,
     $CHISQ,ALAMDA,funcfitnam)
      IF(DABS(CHISQ-OLDCHISQ).LT.1.D-6) THEN
        ICO=ICO+1
      ELSE
        ICO=0
      ENDIF
      IF(ICO.EQ.5) GOTO 16
      OLDCHISQ=CHISQ
c 68   Format('Reduced Chi square = ',f13.8)
 15   Continue
      Write(*,*) 'Max. number of iterations exceeded'

 16   continue

c      write(*,*)'sig',sigf(1),'redchi2',dsqrt(CHISQ/(nco-4))

      ! Modify sig so that red chi2 equals 1

      do 90 i=1,nco
 90    sigf(i)=sigf(i)*dsqrt(CHISQ/(nco-4))


      ! Recompute fit with correct sig
      CALL MRQMIN(XF,YF,sigf,NCO,A,LISTA,NP,COVAR,ALPHA,20,
     $CHISQ,ALAMDA,funcfitnam)
c      write(*,*) 'sig_new',sigf(1),'redchi2_new',dsqrt(CHISQ/(nco-4))

      ! Call MRQMIN one final time with ALAMDA=0.0 so that array COVAR will
      ! return the covariance matrix
      CALL MRQMIN(XF,YF,SIGF,NCO,A,LISTA,NP,COVAR,ALPHA,20,
     $CHISQ,0.D0,funcfitnam)
c      CALL MRQMIN(XF,YF,SIG_cent,NCO,A,LISTA,NP,COVAR,ALPHA,20,
c     $CHISQ,0.D0)

      ! Gaussian parameters uncertainties
      !Derived from the square roots of the diagonal elements of COVAR
      da(1)=dsqrt(covar(1,1))
      da(2)=dsqrt(covar(2,2)) !sig RVmaximum
      da(3)=dsqrt(covar(3,3))
      da(4)=dsqrt(covar(4,4))
      
C      !Gaussian: Y=A(4)+A(1)*dexp(-(X-A(2))**2.d0/A(3))
C      contrast=100.d0*A(1)
C      rvmaximum=A(2)
Cc      FWHM=A(3)
C      fwhm=2*dsqrt(log(2)*A(3))
C      fitmax=A(4)
C      ! FWHM=2*dsqrt(log(2)*A(3))
C      !log: natural

C      write(*,*) A(1),A(2),A(3),A(4)

      return
      END
!----------------------------------------------------------------------


! Levenberg-Marquardt Method
! ==========================
! Subroutines:

! Perform one iteration of Levenberg-Marquardt's method
      SUBROUTINE mrqmin(x,y,sig,ndata,a,ia,ma,covar,alpha,nca,chisq,
     *alamda,funcfitnam)

      ! a: Coefficients of the function to fit
      ! Uses: covsrt,gaussj,mrqcof
      INTEGER ma,nca,ndata,ia(ma),MMAX
      REAL*8 alamda,chisq,a(ma),alpha(nca,nca),covar(nca,nca),
     *sig(ndata),x(ndata),y(ndata)
      !Set to largest number of fit parameters
      PARAMETER (MMAX =20)
      INTEGER j,k,l,mfit
      REAL*8 ochisq,atry(MMAX),beta(MMAX),da(MMAX)
      SAVE ochisq,atry,beta,da,mfit
      character*50 funcfitnam
      if(alamda.lt.0.D0)then !Initialization
        mfit=0
        do 11 j=1,ma
          if (ia(j).ne.0) mfit=mfit+1
11      continue
        alamda=0.001D0
        call mrqcof(x,y,sig,ndata,a,ia,ma,alpha,beta,nca,chisq,
     &   funcfitnam)
        ochisq=chisq
        do 12 j=1,ma
          atry(j)=a(j)
12      continue
      endif
      do 14 j=1,mfit !Alter linearized fitting matrix. by augmenting diagonal elements
        do 13 k=1,mfit
          covar(j,k)=alpha(j,k)
13      continue
        covar(j,j)=alpha(j,j)*(1.D0+alamda)
        da(j)=beta(j)
14    continue
      call gaussj(covar,mfit,nca,da,1,1) !Matrix solution
      if(alamda.eq.0.D0)then ! Once converged, evaluate covariance matrix
        call covsrt(covar,nca,ma,ia,mfit)
        return
      endif
      j=0
      do 15 l=1,ma
        if(ia(l).ne.0) then
          j=j+1
          atry(l)=a(l)+da(j)
        endif
15    continue
      call mrqcof(x,y,sig,ndata,atry,ia,ma,covar,da,nca,chisq,
     & funcfitnam)
      if(chisq.lt.ochisq)then !Success, accept new solution
        alamda=0.1D0*alamda
        ochisq=chisq
        do 17 j=1,mfit
          do 16 k=1,mfit
            alpha(j,k)=covar(j,k)
16        continue
          beta(j)=da(j)
17      continue
        do 18 l=1,ma
          a(l)=atry(l)
18      continue
      else !Failure, increase alamda and return
        alamda=10.D0*alamda
        chisq=ochisq
      endif
      return
      END
!----------------------------------------------------------------------

! Called by routine `mrqmin` to evaluate the linearized fitting matrix `alpha` and vector `beta`, and calculate `chisq`
      SUBROUTINE mrqcof(x,y,sig,ndata,a,ia,ma,alpha,beta,nalp,chisq,
     & funcfitnam)
      INTEGER ma,nalp,ndata,ia(ma),MMAX
      REAL*8 chisq,a(ma),alpha(nalp,nalp),beta(ma),sig(ndata),x(ndata),
     *y(ndata)
      PARAMETER (MMAX=20)
      INTEGER mfit,i,j,k,l,m
      REAL*8 dy,sig2i,wt,ymod,dyda(MMAX)!,z,xm,xe
      character*50 funcfitnam
      mfit=0
      do 11 j=1,ma
        if (ia(j).ne.0) mfit=mfit+1
11    continue
      do 13 j=1,mfit !Initialize (symmetric) alpha, beta
        do 12 k=1,j
          alpha(j,k)=0.D0
12      continue
        beta(j)=0.D0
13    continue
      chisq=0.D0
      do 16 i=1,ndata !Summation loop over all data
        call funcsg(x(i),a,ymod,dyda,ma,funcfitnam)
        sig2i=1.D0/(sig(i)*sig(i))
        dy=y(i)-ymod
        j=0
        do 15 l=1,ma
          if(ia(l).ne.0) then
            j=j+1
            wt=dyda(l)*sig2i
            k=0
            do 14 m=1,l
              if(ia(m).ne.0) then
                k=k+1
                alpha(j,k)=alpha(j,k)+wt*dyda(m)
              endif
14          continue
            beta(j)=beta(j)+dy*wt
          endif
15      continue
        chisq=chisq+dy*dy*sig2i !Find chi2
16    continue
      do 18 j=2,mfit !Fill in the symmetric side
        do 17 k=1,j-1
          alpha(k,j)=alpha(j,k)
17      continue
18    continue
      return
      END
!----------------------------------------------------------------------

! Repack covariance matrix COVAR using the order of parameters in IA
! Input parameters:
!       COVAR - input covariance matrix (double precision)
!       NPC - dimensions of COVAR (integer)
!       MA - total number of parameters in fit (integer)
!       IA - list of parameters in fit selected (integer vector)
!       MFIT - number of parameters selected (integer)
! Output parameters:
!       COVAR - repacked covariance matrix
      SUBROUTINE covsrt(covar,npc,ma,ia,mfit)
      INTEGER ma,mfit,npc,ia(ma)
      DOUBLE PRECISION covar(npc,npc)
      INTEGER i,j,k
      DOUBLE PRECISION swap
      do 12 i=mfit+1,ma
        do 11 j=1,i
          covar(i,j)=0.d0
          covar(j,i)=0.d0
11      continue
12    continue
      k=mfit
      do 15 j=ma,1,-1
        if(ia(j).ne.0)then
          do 13 i=1,ma
            swap=covar(i,k)
            covar(i,k)=covar(i,j)
            covar(i,j)=swap
13        continue
          do 14 i=1,ma
            swap=covar(k,i)
            covar(k,i)=covar(j,i)
            covar(j,i)=swap
14        continue
          k=k-1
        endif
15    continue
      return
      END
!----------------------------------------------------------------------

! Linear equation solution by Gauss-Jordan elimination
      SUBROUTINE gaussj(a,n,np,b,m,mp)
      INTEGER m,mp,n,np,NMAX
      DOUBLE PRECISION a(np,np),b(np,mp)
      PARAMETER (NMAX=50)
      INTEGER i,icol,irow,j,k,l,ll,indxc(NMAX),indxr(NMAX),ipiv(NMAX)
      DOUBLE PRECISION big,dum,pivinv
      do 11 j=1,n
        ipiv(j)=0
11    continue
      do 22 i=1,n
        big=0.
        do 13 j=1,n
          if(ipiv(j).ne.1)then
            do 12 k=1,n
              if (ipiv(k).eq.0) then
                if (abs(a(j,k)).ge.big)then
                  big=abs(a(j,k))
                  irow=j
                  icol=k
                endif
              else if (ipiv(k).gt.1) then
c               pause 'singular matrix in gaussj'
              endif
12          continue
          endif
13      continue
        ipiv(icol)=ipiv(icol)+1
        if (irow.ne.icol) then
          do 14 l=1,n
            dum=a(irow,l)
            a(irow,l)=a(icol,l)
            a(icol,l)=dum
14        continue
          do 15 l=1,m
            dum=b(irow,l)
            b(irow,l)=b(icol,l)
            b(icol,l)=dum
15        continue
        endif
        indxr(i)=irow
        indxc(i)=icol
c       if (a(icol,icol).eq.0.d0) pause 'singular matrix in gaussj'
        pivinv=1./a(icol,icol)
        a(icol,icol)=1.d0
        do 16 l=1,n
          a(icol,l)=a(icol,l)*pivinv
16      continue
        do 17 l=1,m
          b(icol,l)=b(icol,l)*pivinv
17      continue
        do 21 ll=1,n
          if(ll.ne.icol)then
            dum=a(ll,icol)
            a(ll,icol)=0.d0
            do 18 l=1,n
              a(ll,l)=a(ll,l)-a(icol,l)*dum
18          continue
            do 19 l=1,m
              b(ll,l)=b(ll,l)-b(icol,l)*dum
19          continue
          endif
21      continue
22    continue
      do 24 l=n,1,-1
        if(indxr(l).ne.indxc(l))then
          do 23 k=1,n
            dum=a(k,indxr(l))
            a(k,indxr(l))=a(k,indxc(l))
            a(k,indxc(l))=dum
23        continue
        endif
24    continue
      return
      END
! ---------------------------------------------------------------------

! Functions to fit
! ----------------

      SUBROUTINE FUNCSG(X,A,Y,DYDA,NA,funcfitnam)
      IMPLICIT REAL*8(A-H,O-Z)
      DIMENSION A(NA),DYDA(NA)
      character*50 funcfitnam !Name of the function to fit
      ! Possible functions to fit: gaussian, sinc

      ! --- Gaussian
      if(funcfitnam.eq.'gaussian') then
c       print*,'Fitting Gaussian'
       Y=A(4)+A(1)*dexp(-(X-A(2))**2.d0/A(3)) 
       ! Derivatives
       DYDA(1)=dexp(-(X-A(2))**2.d0/A(3))
       DYDA(2)=2.d0*((X-A(2))/A(3))*A(1)*dexp(-(X-A(2))**2.d0/A(3))
       DYDA(3)=(((X-A(2))/A(3))**2.d0)*A(1)*dexp(-(X-A(2))**2.d0/A(3))
       DYDA(4)=1.d0

      ! --- sinc
      ! sinc(x)=shift + amp * sin((x-cen)/wid)/((x-cen)/wid)
      ! a(1)=amp, a(2)=cen, a(3)=wid, a(4)=shift
      elseif(funcfitnam.eq.'sinc') then
c       print*,'Fitting sinc'
       if(x-a(2).eq.0) then
        y=a(4)+a(1)
       elseif(x.ne.0) then
        y=a(4) + a(1)*sin((x-a(2))/a(3)) / ((x-a(2))/a(3))

        ! Revisar NaN
        dyda(1)=sin((x-a(2))/a(3)) / ((x-a(2))/a(3))
        dyda(2)=-a(1)*cos((x-a(2))/a(3)) / (a(3)*(x-a(2))/a(3))
     &   + a(1)*sin((x-a(2)) / a(3))/a(3)
        dyda(3)=-a(1)*cos((x-a(2))/a(3))*(x-a(2)) / 
     &   ((a(3)**2)*(x-a(2))/a(3)) + a(1)*sin((x-a(2))/a(3)) * 
     &   (x-a(2)) / (a(3)**2)
        dyda(4)=1.d0

       else 
        print*,'No fit'
        stop
      endif

      endif

      RETURN
      END
! ---------------------------------------------------------------------

! Compute Bisector
      subroutine bisector(nrc,rdw,ccx,bspan,fitmax)
      implicit double precision (a-h,o-z)
      ! Input
      integer,intent(in) :: nrc !=nrv
      double precision,dimension(nrc),intent(in) :: rdw !=rv
      double precision,dimension(nrc),intent(in) :: ccx !=ccf
      double precision,intent(in) :: fitmax !=a(4)
      ! Output
      double precision,intent(out) :: bspan
      ! Local
      dimension ccfit(100),rvfit(100)

Cf2py integer,intent(in,hide) :: nrc

c      ccfmn=1.d0
      ccfmn=fitmax
      ccfal=0.d0
      ccfar=0.d0

      ! Find CCF minimum and maxima
      do i=1,nrc
       if(ccx(i).lt.ccfmn) then
        ccfmn=ccx(i)
        rvmax=rdw(i)
       endif
      enddo
      do i=1,nrc
       if(ccx(i).gt.ccfal.and.rdw(i).lt.rvmax) then
        ccfal=ccx(i)
        rvmaxl=rdw(i)
       endif
       if(ccx(i).gt.ccfar.and.rdw(i).gt.rvmax) then
        ccfar=ccx(i)
        rvmaxr=rdw(i)
       endif
      enddo
      ccfmx=fitmax

      szon=200.d0
      spre=2.d0
      do i=1,100
       ccfit(i)=ccfmn+spre*((ccfmx-ccfmn)/szon)+((szon-spre-1.d0)/(99.d
     *0*szon))*(ccfmx-ccfmn)*(i-1)
       do j=1,nrc
        if(ccx(j).gt.ccfit(i).and.ccx(j+1).lt.ccfit(i).and.rdw(j).lt.
     *rvmax.and.rdw(j).gt.rvmaxl) then
       rv1=rdw(j)+(rdw(j+1)-rdw(j))*(ccfit(i)-ccx(j))/(ccx(j+1)-ccx(j))
        endif
        if(ccx(j).lt.ccfit(i).and.ccx(j+1).gt.ccfit(i).and.rdw(j).gt.
     *rvmax.and.rdw(j).lt.rvmaxr) then
       rv2=rdw(j)+(rdw(j+1)-rdw(j))*(ccfit(i)-ccx(j))/(ccx(j+1)-ccx(j))
        endif
       enddo
       rvfit(i)=(rv2+rv1)/2.d0
      enddo
      close(1)

c Bisector span zone limits
      ccf10=ccfmx-(ccfmx-ccfmn)*0.1d0
      ccf40=ccfmx-(ccfmx-ccfmn)*0.4d0
      ccf60=ccfmx-(ccfmx-ccfmn)*0.6d0
      ccf90=ccfmx-(ccfmx-ccfmn)*0.9d0

      bsp1=0.d0
      bsp2=0.d0
      ns1=0
      ns2=0
      do i=1,100
       if(ccfit(i).lt.ccf10.and.ccfit(i).gt.ccf40) then
        bsp1=bsp1+rvfit(i)
        ns1=ns1+1
       elseif(ccfit(i).lt.ccf60.and.ccfit(i).gt.ccf90) then
        bsp2=bsp2+rvfit(i)
        ns2=ns2+1
       endif
      enddo
      bsp1=bsp1/dfloat(ns1)
      bsp2=bsp2/dfloat(ns2)
      bspan=bsp2-bsp1

      return
      end
! ---------------------------------------------------------------------

! Compute Bisector
! Return bisector, not only bisector span
      subroutine bisector2(nrv,rv,ccf,fitmax,bspan,rvfit,ccfit)
      implicit double precision (a-h,o-z)
      ! Input
      integer,intent(in) :: nrv
      double precision,dimension(nrv),intent(in) :: rv
      double precision,dimension(nrv),intent(in) :: ccf
      double precision,intent(in) :: fitmax !=a(4)
      ! Output
      double precision,intent(out) :: bspan
      double precision,dimension(100),intent(out) :: ccfit,rvfit

Cf2py integer,intent(in,hide) :: nrv

c      ccfmn=1.d0
      ccfmn=fitmax
      ccfal=0.d0
      ccfar=0.d0

      ! Find CCF minimum and maxima
      do i=1,nrv
       if(ccf(i).lt.ccfmn) then
        ccfmn=ccf(i)
        rvmax=rv(i)
       endif
      enddo
      do i=1,nrv
       if(ccf(i).gt.ccfal.and.rv(i).lt.rvmax) then
        ccfal=ccf(i)
        rvmaxl=rv(i)
       endif
       if(ccf(i).gt.ccfar.and.rv(i).gt.rvmax) then
        ccfar=ccf(i)
        rvmaxr=rv(i)
       endif
      enddo
      ccfmx=fitmax

      szon=200.d0
      spre=2.d0
      do i=1,100
       ccfit(i)=ccfmn+spre*((ccfmx-ccfmn)/szon)+((szon-spre-1.d0)/(99.d
     *0*szon))*(ccfmx-ccfmn)*(i-1)
       do j=1,nrv
        if(ccf(j).gt.ccfit(i).and.ccf(j+1).lt.ccfit(i).and.rv(j).lt.
     *rvmax.and.rv(j).gt.rvmaxl) then
       rv1=rv(j)+(rv(j+1)-rv(j))*(ccfit(i)-ccf(j))/(ccf(j+1)-ccf(j))
        endif
        if(ccf(j).lt.ccfit(i).and.ccf(j+1).gt.ccfit(i).and.rv(j).gt.
     *rvmax.and.rv(j).lt.rvmaxr) then
       rv2=rv(j)+(rv(j+1)-rv(j))*(ccfit(i)-ccf(j))/(ccf(j+1)-ccf(j))
        endif
       enddo
       rvfit(i)=(rv2+rv1)/2.d0
      enddo
      close(1)

c Bisector span zone limits
      ccf10=ccfmx-(ccfmx-ccfmn)*0.1d0
      ccf40=ccfmx-(ccfmx-ccfmn)*0.4d0
      ccf60=ccfmx-(ccfmx-ccfmn)*0.6d0
      ccf90=ccfmx-(ccfmx-ccfmn)*0.9d0

      bsp1=0.d0
      bsp2=0.d0
      ns1=0
      ns2=0
      do i=1,100
       if(ccfit(i).lt.ccf10.and.ccfit(i).gt.ccf40) then
        bsp1=bsp1+rvfit(i)
        ns1=ns1+1
       elseif(ccfit(i).lt.ccf60.and.ccfit(i).gt.ccf90) then
        bsp2=bsp2+rvfit(i)
        ns2=ns2+1
       endif
      enddo
      bsp1=bsp1/dfloat(ns1)
      bsp2=bsp2/dfloat(ns2)
      bspan=bsp2-bsp1

      return
      end
! ---------------------------------------------------------------------

! Sort an array (min to max)
      SUBROUTINE sort(n,arr)
      INTEGER n,M,NSTACK
      DOUBLE PRECISION arr(n)
      PARAMETER (M=7,NSTACK=50)
      INTEGER i,ir,j,jstack,k,l,istack(NSTACK)
      DOUBLE PRECISION a,temp
      jstack=0
      l=1
      ir=n
1     if(ir-l.lt.M)then
        do 12 j=l+1,ir
          a=arr(j)
          do 11 i=j-1,l,-1
            if(arr(i).le.a)goto 2
            arr(i+1)=arr(i)
11        continue
          i=l-1
2         arr(i+1)=a
12      continue
        if(jstack.eq.0)return
        ir=istack(jstack)
        l=istack(jstack-1)
        jstack=jstack-2
      else
        k=(l+ir)/2
        temp=arr(k)
        arr(k)=arr(l+1)
        arr(l+1)=temp
        if(arr(l).gt.arr(ir))then
          temp=arr(l)
          arr(l)=arr(ir)
          arr(ir)=temp
        endif
        if(arr(l+1).gt.arr(ir))then
          temp=arr(l+1)
          arr(l+1)=arr(ir)
          arr(ir)=temp
        endif
        if(arr(l).gt.arr(l+1))then
          temp=arr(l)
          arr(l)=arr(l+1)
          arr(l+1)=temp
        endif
        i=l+1
        j=ir
        a=arr(l+1)
3       continue
          i=i+1
        if(arr(i).lt.a)goto 3
4       continue
          j=j-1
        if(arr(j).gt.a)goto 4
        if(j.lt.i)goto 5
        temp=arr(i)
        arr(i)=arr(j)
        arr(j)=temp
        goto 3
5       arr(l+1)=arr(j)
        arr(j)=a
        jstack=jstack+2
        if(ir-i+1.ge.j-l)then
          istack(jstack)=ir
          istack(jstack-1)=i
          ir=j-1
        else
          istack(jstack)=j-1
          istack(jstack-1)=l
          l=i
        endif
      endif
      goto 1
      END
C  (C) Copr. 1986-92 Numerical Recipes Software &H1216.
!----------------------------------------------------------------------

CC Linear interpolation
C      subroutine interplin(x,y,n,xnew,ynew)
C      implicit none
C      double precision,dimension(n) :: x,y !Original grid
C      integer :: n
C      double precision :: xnew,ynew !Point to interpolate
C      integer :: idx1,idx2 !Indices of x points closest to xnew
C                           !x(idx1)<=xnew, x(idx2)>xnew
C      double precision :: dx1,minterp,ninterp
C      integer :: j
C
C      ! Indices of x points closest to xnew
C      if(x(1).gt.xnew) then 
C       ynew=0.d0
C      elseif(x(n).lt.xnew) then 
C       ynew=0.d0
C      else
C       ynew=999999.d0
C       dx1=999999.d0
C       idx1=n
C       idx2=1
C       !x(idx1)<=xnew
C       do j=1,n
C        if((xnew-x(j)).ge.0.d0) then
C         !Update idx1 if x(j) closer than previous point
C         if((xnew-x(j)).lt.dx1) then
C          dx1=xnew-x(j)
C          idx1=j
C         endif
C        else
C         exit
C        endif
C       enddo
C       !x(idx2)>xnew
C       idx2=idx1+1
C       ! Line
C       minterp=(y(idx2)-y(idx1))/(x(idx2)-x(idx1))
C       ninterp=y(idx1)-minterp*x(idx1)
C       ! Interpolate
C       ynew=minterp*xnew+ninterp
C      endif
C      end
CC--------------------------------------------------------------------

C Interpolate Numerical Recipes
      subroutine interplin(x,y,n,xnew,ynew,dynew)
      implicit none
      integer :: n
      double precision,intent(in),dimension(n) :: x,y !Original grid
      double precision,intent(in) :: xnew !Point to interpolate
      double precision,intent(out) :: ynew,dynew !Point to interpolate
      integer :: m
      integer :: j
      integer :: k
c      integer :: idx1,idx2 !Indices of x points closest to xnew
c                           !x(idx1)<=xnew, x(idx2)>xnew
c      double precision :: dx1,minterp,ninterp
c      integer :: j

      m=2
      j=-1
      call locate(x,n,xnew,j)
      k = min(max(j-(m-1)/2,1),n+1-m)
      call polint(x(k:k+m),y(k:k+m),m,xnew,ynew,dynew)
      return
      end
C--------------------------------------------------------------------

C Numerical Recipes
      SUBROUTINE locate(xx,n,x,j)
      INTEGER j,n
      double precision x
      double precision xx(n)
      INTEGER jl,jm,ju
      jl=0
      ju=n+1
10    if(ju-jl.gt.1)then
        jm=(ju+jl)/2
        if((xx(n).ge.xx(1)).eqv.(x.ge.xx(jm)))then
          jl=jm
        else
          ju=jm
        endif
      goto 10
      endif
      if(x.eq.xx(1))then
        j=1
      else if(x.eq.xx(n))then
        j=n-1
      else
        j=jl
      endif
      return
      END
C--------------------------------------------------------------------

C Numerical Recipes
C Given  arrays xa and ya, each of length n, and given a value x, this 
C routine returns a value y, and an error estimate dy. If P(x) is the 
C polynomial of degree N-1 such that P(xa_i)==ya_i, i=1,...,n, then 
C the returned value y=P(x).
      SUBROUTINE polint(xa,ya,n,x,y,dy)
      INTEGER n,NMAX
      double precision dy,x,y,xa(n),ya(n)
c      PARAMETER (NMAX=5000)
      PARAMETER (NMAX=10)
      INTEGER i,m,ns
      double precision den,dif,dift,ho,hp,w,c(NMAX),d(NMAX)
      ns=1
      dif=abs(x-xa(1))
      ! Find the index ns of the closest table entry
      do 11 i=1,n
        dift=abs(x-xa(i))
        if (dift.lt.dif) then
          ns=i
          dif=dift
        endif
        ! Initialize the tableau of c's and d's
        c(i)=ya(i)
        d(i)=ya(i)
11    continue
      y=ya(ns) !Initial approximation to y
      ns=ns-1
      ! For each column of the tableau, loop over the current c's and
      ! d'a and update them
      do 13 m=1,n-1
        do 12 i=1,n-m
          ho=xa(i)-x
          hp=xa(i+m)-x
          w=c(i+1)-d(i)
          den=ho-hp
          ! Error if two input xa's are (to within roundoff) identical
          if(den.eq.0.)pause 'failure in polint'
          den=w/den
          ! Update c's and d's
          d(i)=hp*den
          c(i)=ho*den
12      continue
        ! After each column in the tableau is completed, we decide 
        ! which correction, c or d, we want to add to our 
        ! accumulating value of y, i.e., which path to take through 
        ! the tableau-forking up or down. We do this in such a way as
        ! to take the most 'straight line' route through the tableau to
        ! its apex, updating ns accordingly to keep track of where we 
        ! are. This route keeps the partical approximations centered 
        ! (insofar as possible) on the target x. The last dy added is 
        ! thus the error indication.
        if (2*ns.lt.n-m)then
          dy=c(ns+1)
        else
          dy=d(ns)
          ns=ns-1
        endif
        y=y+dy
C        print*,'m,i',m,i,'y,dy',y,dy
13    continue
C      print*,'final y,dy',y,dy
      return
      END
C--------------------------------------------------------------------


C Spline interpolation
C x,y: Input
C yp1,ypn: First derivative of the interpolating function at values 1 and n. If larger than 1e30, routine sets boundary condition for a natural spline, i.e. with zero second derivative on that boundary.
C y2: Second derivatives of the interpolating function at tabulated points x
      SUBROUTINE spline(x,y,n,yp1,ypn,y2)
      INTEGER n,NMAX
      double precision yp1,ypn,x(n),y(n),y2(n)
C      PARAMETER (NMAX=500)
c      PARAMETER (NMAX=5000)
      PARAMETER (NMAX=500000)
      INTEGER i,k
      double precision p,qn,sig,un,u(NMAX)
      if (yp1.gt..99e30) then
        y2(1)=0.
        u(1)=0.
      else
        y2(1)=-0.5
        u(1)=(3./(x(2)-x(1)))*((y(2)-y(1))/(x(2)-x(1))-yp1)
      endif
      do 11 i=2,n-1
        sig=(x(i)-x(i-1))/(x(i+1)-x(i-1))
        p=sig*y2(i-1)+2.
        y2(i)=(sig-1.)/p
        u(i)=(6.*((y(i+1)-y(i))/(x(i+
     *1)-x(i))-(y(i)-y(i-1))/(x(i)-x(i-1)))/(x(i+1)-x(i-1))-sig*
     *u(i-1))/p
11    continue
      if (ypn.gt..99e30) then
        qn=0.
        un=0.
      else
        qn=0.5
        un=(3./(x(n)-x(n-1)))*(ypn-(y(n)-y(n-1))/(x(n)-x(n-1)))
      endif
      y2(n)=(un-qn*u(n-1))/(qn*y2(n-1)+1.)
      do 12 k=n-1,1,-1
        y2(k)=y2(k)*y2(k+1)+u(k)
12    continue
      return
      END

C xa,ya: Input
C y2a: Output from spline
C x: Value for which the spline is evaluated
C y: Output
      SUBROUTINE splint(xa,ya,y2a,n,x,y)
      INTEGER n
      double precision x,y,xa(n),y2a(n),ya(n)
      INTEGER k,khi,klo
      double precision a,b,h
      klo=1
      khi=n
1     if (khi-klo.gt.1) then
        k=(khi+klo)/2
        if(xa(k).gt.x)then
          khi=k
        else
          klo=k
        endif
      goto 1
      endif
      h=xa(khi)-xa(klo)
      if (h.eq.0.) pause 'bad xa input in splint'
      a=(xa(khi)-x)/h
      b=(x-xa(klo))/h
      y=a*ya(klo)+b*ya(khi)+((a**3-a)*y2a(klo)+(b**3-b)*y2a(khi))*(h**
     *2)/6.
      return
      END
