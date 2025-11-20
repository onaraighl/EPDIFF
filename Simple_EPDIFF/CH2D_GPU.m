function [x,y,X,Y,u_hist,v_hist,t_vec,energy_vec] = CH2D_GPU()

% *************************************************************************
% 2DCH / EPDiff pseudospectral code.
% October 2025
% Lennon Ó Náraigh (and ChatGPT)
% 
% Inputs: null
% Outputs: the mesh [X,Y] and the velocity components u and v at the final
% time T.
%
% KEY POINTS:
%
% * Final time T is controlled below under "numerical parameters".
% * Intermediate u and v values are printed to the screen every snapEvery
%   timesteps, where snapEvery is controlled below also under "numerical
%   parameters".
% * FFTs and elementwise operations have been despatched to NVIDIA GPUs 
%   simply by declaring special arrays at the start of the computation.

% *************************************************************************
% Technical:
%
% TEMPORAL DISCRETIZATION:
% RK4, hence EXPLICIT
%
% SPATIAL DERIVATIVES:
% Computed using Fourier transforms, hence psuedospectral method.
%
% REFERENCES:
% 1.    Holm, D.D. and Staley, M.F., 2013. Interaction dynamics of singular 
%       wave fronts. arXiv preprint arXiv:1301.1460.
% 2.    Chertock, A., Du Toit, P. and Marsden, J.E., 2012. 
%       Integration of the EPDiff equation by particle methods.
%       ESAIM: Mathematical Modelling and Numerical Analysis, 46(3), pp.515-534.
% 3.    Azencot, O., Vantzos, O. and Ben‐Chen, M., 2018, August. An explicit 
%       structure‐preserving numerical scheme for EPDiff. 
%       In Computer Graphics Forum (Vol. 37, No. 5, pp. 107-119).

% *************************************************************************
% Physical parameters:

sigma=0.2;

alpha=sigma;
alphasq=alpha^2;

% Generalizatoin parameter, beta=1 is CH
beta=1;

% *************************************************************************
% Numerical parameters

Lx=5;
Ly=5;

Nx=256;
Ny=256;

dx = Lx/Nx; % grid spacing
dy = Ly/Ny; % grid spacing

T=10; % final time
dt=1e-4; % timestep

Nt=ceil(T/dt);

t_val=0; % time value, initially at zero.
snapEvery=1000; % snapshot plotted every snapEvery timesteps
snapI=0; % snapshot counter

% vector of energy values
t_vec=0*(1:Nt);
energy_vec=0*(1:Nt);

% *************************************************************************
% Numerical preliminaries:

x = (0:Nx-1)*dx;
y = (0:Ny-1)*dy;

% Centred coordinate system: 
x=x-(Lx/2);
y=y-(Ly/2);

% Create mesh in real space:
[X,Y] = meshgrid(x,y); 

% Wavvenumber grids (from ChatGPT):
kx = (2*pi/Lx) * [0:(Nx/2-1), -Nx/2:-1];
ky = (2*pi/Ly) * [0:(Ny/2-1), -Ny/2:-1];
[KX, KY] = meshgrid(kx, ky);   % note: meshgrid gives (rows = y), consistent with fft2 ordering

% Other operators:
K2 = KX.^2 + KY.^2;
Darryl_op=(1+alphasq*K2).^(-1);
Darryl_op_sigma=(1+(sigma^2)*K2).^(-1);

% Hou-Li filter -- Doesn't seem to do the business at all for EPDiff.
K = sqrt(KX.^2 + KY.^2); 
Kmax = max(K(:));  
p = 36;         % choose 18, 36, 72, etc.
alphaf = 36;     % typical strength
filter2D = exp( -alphaf * (K/Kmax).^p );

% % Instead, use standard de-aliasing instead, with 2/3 rule cutoff
% kx_cut = (2*pi/Lx)* (2/3) * (Nx/2);
% ky_cut = (2*pi/Ly)* (2/3) * (Ny/2);

% Dealiasing mask
mask = filter2D;%(abs(KX) < kx_cut) & (abs(KY) < ky_cut);

% *************************************************************************
% Initial conditions
% *************************************************************************


% % Option 1: A single momentum strip
% 
% xc1=-1;
% yc1=0;
% wy=1;
% 
% small=2*dx;
% Mx = 2*((2*pi*small*small)^(-1/2))*exp(-(X-xc1).*(X-xc1)/(2*small^2)).*double( abs(Y-yc1)<=wy/2);

% *************************************************************************
% Option 2: Two momentum strips

xc1=-0.75;
yc1=0.25;
wy=1;

xc2=-0.25;
yc2=-0.25;

small=2*dx;
Mx1 = 2*((2*pi*small*small)^(-1/2))*exp(-(X-xc1).*(X-xc1)/(2*small^2)).*double( abs(Y-yc1)<=wy/2);
Mx2 =   ((2*pi*small*small)^(-1/2))*exp(-(X-xc2).*(X-xc2)/(2*small^2)).*double( abs(Y-yc2)<=wy/2);
Mx=Mx1+Mx2;

% *************************************************************************
% Finalize initial conditions:

Mx_hat=fft2(Mx);
Mx_hat=Darryl_op_sigma.*Mx_hat;
Mx=real(ifft2(Mx_hat));

u_init=Mx/(max(Mx(:))+1e-10);
v_init=0*X;

u = u_init;
v = v_init;

uhat=fft2(u); 
vhat=fft2(v);

% *************************************************************************
% *************************************************************************
% Move arrays to GPU (ChatGPT):

KX = gpuArray(KX);
KY = gpuArray(KY);
u = gpuArray(u);
v = gpuArray(v);
uhat = gpuArray(uhat);
vhat = gpuArray(vhat);

K2 = gpuArray(K2);
Darryl_op=gpuArray(Darryl_op);
% filter2D=gpuArray(filter2D);
mask=gpuArray(mask);

% *************************************************************************



for t_ctr=1:Nt

    % RK4 update:
    [k1u, k1v] = RHS_RK4(uhat,              vhat             );
    [k2u, k2v] = RHS_RK4(uhat + 0.5*dt*k1u, vhat + 0.5*dt*k1v);
    [k3u, k3v] = RHS_RK4(uhat + 0.5*dt*k2u, vhat + 0.5*dt*k2v);
    [k4u, k4v] = RHS_RK4(uhat +     dt*k3u, vhat +     dt*k3v);
    
    % update Fourier coefficients
    uhat = uhat+ (dt/6)*(k1u + 2*k2u + 2*k3u + k4u);
    vhat = vhat + (dt/6)*(k1v + 2*k2v + 2*k3v + k4v);

    % transfor back to real space for visualization
    u=real(ifft2(uhat)); v=real(ifft2(vhat));

    % keep track of time - literally!
    t_val=t_val+dt;

    % Plot and monitor output:
    if mod(t_ctr,snapEvery)==0

        snapI=snapI+1;

        % Bring results back to CPU
        u_cpu = gather(u);
        v_cpu = gather(v);

        % gather results to space-time array.
        u_hist(snapI,:,:)=u_cpu; v_hist(snapI,:,:)=v_cpu; 

        % output results to screen
        clf;
        imagesc(x,y,sqrt(u_cpu.^2+v_cpu.^2)); axis xy equal tight;
        title(strcat('|u|, t=',num2str(t_val))); colorbar;
        xlabel('x')
        ylabel('y')
        drawnow;

        display(t_val)
    end

    % Monitor energy
    energy_t=get_energy(uhat,vhat);
    t_vec(t_ctr)=t_val;
    energy_vec(t_ctr)=energy_t;
end

% Bring results back to CPU
u = gather(u);
v = gather(v);

u_hist=u;
v_hist=v;

% *************************************************************************
% RHS calculation for RK4 
% "L" for local in each of the variable names.

function [RHS_x_hat, RHS_y_hat] = RHS_RK4(u_hatL,v_hatL) 
    % transform to physical space:
    uL = real(ifft2(u_hatL)); 
    vL = real(ifft2(v_hatL));

    % spectral derivatives
    u_xL  =  real(ifft2(1i*KX .*u_hatL));
    u_yL  =  real(ifft2(1i*KY .*u_hatL));
    lap_uL = -real(ifft2(K2.*u_hatL));

    v_xL  =  real(ifft2(1i*KX .*v_hatL));
    v_yL  =  real(ifft2(1i*KY .*v_hatL));
    lap_vL = -real(ifft2(K2.*v_hatL));

    % divergence:
    divu=u_xL+v_yL;
    
    % convective derivatives:
    mxL=uL-alphasq*lap_uL;
    myL=vL-alphasq*lap_vL;

    mx_hat=fft2(mxL);
    dx_mx=real(ifft2(1i*KX.*mx_hat));
    dy_mx=real(ifft2(1i*KY.*mx_hat));

    my_hat=fft2(myL);
    dx_my=real(ifft2(1i*KX.*my_hat));
    dy_my=real(ifft2(1i*KY.*my_hat));

    conv_x=uL.*dx_mx+vL.*dy_mx;
    conv_y=uL.*dx_my+vL.*dy_my;

    % Note - this contraction was wrong.
    % stretch_x=mx.*u_xL + my.*u_yL;
    % stretch_y=mx.*v_xL + my.*v_yL;

    stretch_x=mxL.*u_xL + myL.*v_xL;
    stretch_y=mxL.*u_yL + myL.*v_yL;

    expand_x=mxL.*divu*(beta-1);
    expand_y=myL.*divu*(beta-1);

    RHS_x=-conv_x-stretch_x-expand_x;
    RHS_y=-conv_y-stretch_y-expand_y;

    % *********************************************************************
    % Construct final RHS in Fourier space
    RHS_x_hat = fft2(RHS_x);
    RHS_y_hat = fft2(RHS_y);

    % Apply filter
    RHS_x_hat = mask .* RHS_x_hat;
    RHS_y_hat = mask .* RHS_y_hat;

    % Apply inverse Helmholtz
    RHS_x_hat = Darryl_op .* RHS_x_hat;
    RHS_y_hat = Darryl_op .* RHS_y_hat;
   
end

% *************************************************************************
% Energy calculator

    function et=get_energy(u_hatL,v_hatL)

        % transform to physical space:
        uL = real(ifft2(u_hatL)); 
        vL = real(ifft2(v_hatL));

        % spectral derivatives
        lap_uL = -real(ifft2(K2.*u_hatL));
        lap_vL = -real(ifft2(K2.*v_hatL));

        % convective derivatives:
        mxL=uL-alphasq*lap_uL;
        myL=vL-alphasq*lap_vL;

        % energy estimate:

        et=0.5*sum(sum(mxL.*uL+myL.*vL))*dx*dy;

    end

end






