function DPOAEs_model_Cecilia_Casarini()

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This Matlab function represents a physical model of the cochlea inspired
% by the work of A.Moleti(2009). It also simulates distortion product
% emissions (DPOAEs)by assigning a nonzero value to the parameter roughness.
% The time required to run the code grows with N, with frequency and with duration,
% therefore it is convenient to start with low N values (e.g. N=100) to
% have a first understanding of the model. Reliable results are
% nevertheless obtained with higher values (N=1000).
% The user is supposed to choose the number of cochlear partitions N,
% the duration of the simulation tEnd, the
% input frequency f0, the amplitude of the signal a in dB (that will be
% converted automatically in Pascals) and the roughness parameter.
% The matrix Y is saved by the code and it will contain the displacement in
% its even columns and the velocity of the BM in its odd columns. The lines
% represent the time of the simulation.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all
clc

% Start of time measurement of the code
tic;

%%%%%%%%%%%%%%%%%%% MECHANICAL AND PHYSICAL PARAMETERS %%%%%%%%%%%%%%%%%%%

% Timing of the simulation (s)
tEnd = 0.02;

% Input Frequencies (Hz)
f0 = 2000;
f1 = f0*1.22;
fdp = 2*f0-f1;

%Number of Micromechanical Elements +2 boundary conditions
N = 100;

% Amplitude in dB
adB = 80;

% Convertion to Pascals
aPa = 10^(adB/20)*0.00002;

% Roughness parameter
roughness = 0.6;

% Time vector to be given to the ode45 solver
tspan = [0,tEnd];

% Length of the BM (m)
L = 3.5*(10^(-2));

% Fluid density (Kg*m^(-3))
rho = 10^3;

% Lenght of each cochlear partition (m)
delta = L/(N-2);

% Height (m)
H = 0.001;

% Greenwood map frequency coefficient (s^(-1))
omega0 = 2.0655*(10^4)*2*pi;

% Greenwood map inverse length scale (m^(-1))
k_w = 1.382*(10^2);

% Effective middle ear - oval window damping (S^(-1))
gamma_ow = 5*(10^3);

% Effective middle ear - oval window density (Kg*m^(-2))
sigma_ow = 2;

% BM density (Kg*m^8-2))
sigma_bm = 5.5*(10^(-2));

% Effective middle ear-oval window stiffness ( N*m^(-3))
K_ow = 2*(10^8);

% Middle ear frequency
omega_ow = sqrt(K_ow/sigma_ow); % = 10000

% middle ear mechanical gain of the ossicles
G_me = 21.4;

% OHC non-local interaction range (squared) (m^2)
lambda = 1.2*10^(-7);

% Integer number to be used in the nonlinear mass matrix
K = 1;

% Greenwood equations:

% 1) Place-frequency map
omega_bm = omega0 * exp(-k_w * ((1:(N-2))*delta));

% Expected cochlear places
position1 = (-(log((f0*2*pi)/omega0))/(k_w))*(N-2)/L;
position1=round(position1);
position2 = (-(log((f1*2*pi)/omega0))/(k_w))*(N-2)/L;
position2=round(position2);
position3 = (-(log((fdp*2*pi)/omega0))/(k_w))*(N-2)/L;
position3=round(position3);

% 2) Passive linear damping
% Tuning parameter
Q = 8;
gamma_bm = omega_bm/Q;


%%%%%%%%%%%%%%%%%%%%%%%%%% MATRICES CALCULATION %%%%%%%%%%%%%%%%%%%%%%%%%%

% F Matrix
F = zeros(N,N);
F(1,1) = -delta/H;
F(1,2) = delta/H;
F(N,N) = -2*rho*(delta^2)/H;

i=1;
for j=2:N-1
    F(j,i)= 1;
    F(j, i+1) = -2;
    F(j, i+2) = 1;
    i = i+1;
end

F = H/((2*rho)*(delta^2))*F;
F = sparse(F); % keeps only the non-zero components (to speed up the code)

% A matrix
A = cell(1, N);
A{1} = [-gamma_ow,-(omega_ow)^2; 1, 0];
A{N} = [0, 0; 0, 0];

for n = 2:N-1
    A{n} = [-gamma_bm(n-1),(-(omega_bm(n-1))^2)*(1+roughness*(rand(1)-0.5)); 1, 0];
end

% Ae matrix
Ae = blkdiag(A{1:N});
Ae = sparse(Ae);

% B matrix
B = cell(1,N);

B{1} = [1/sigma_ow; 0];
B{N} = [0;0];

for n = 2:N-1
    B{n} = [1/sigma_bm; 0];
end


% Be matrix
Be = blkdiag(B{1:N});
Be = sparse(Be);
% C matrices
C = cell(1,N);

for n = 1:N
    C{n} = [1,0];
end

% Ce matrix

Ce = blkdiag(C{1:N});
Ce = sparse(Ce);

% I
I = eye(2*N);
I = sparse(I);

% S(t)
S = zeros(N,1);
S(1,1) = G_me;
S = sparse(S);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ODE SOLVER %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
initial_conditions = zeros(2*N,1);
options = odeset('mass', @Mnonlin_func);

[T,Y]=ode45(@DoubleDOF_1stOrderSystem,tspan,initial_conditions,options);
size(Y)

%%%%%%%%%%%%%%%%%%%%%%%%%%% SIMULATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Matrix containing the BM displacement in time
y_displacement = Y(1:size(Y,1), 4:2:2*N-2);
[Maxvalue, Index] = max(max(y_displacement));

% chosen cochlear place evolution in time.
% Change position1 with another number to obtain the spectrum of the signal
% in a different place. If a low number of partitions is used position1
% could not be accurate, therefore try different values until f1 is the
% highest component in the spectrum.
cp = Y(:,position1*2);

% Number of samples to include in the spectrum
fft_length = size(Y,1);
P2 = abs(fft(cp)/fft_length);
P1 = P2(1:fft_length/2+1);
P1(2:end-1) = 2*P1(2:end-1);
Fs=size(Y,1)/0.02;
f = Fs*(0:(fft_length/2))/fft_length;
plot(f,P1)
title('Spectrum of the signal in f1 cochlear place')
xlabel('f (Hz)')
ylabel('Magnitude')
axis([0 5000 0 +inf])
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% MAIN FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Nested function that sets up the 1st order system to be used by ODE solver
    function Zdot = DoubleDOF_1stOrderSystem(t,Z)
        
        %%%%%%%%%%%%%% Main state space system equation %%%%%%%%%%%%%%
        Zdot = Ae*Z + Be*(S*(aPa*(sin(2*pi*f0*t)+sin(2*pi*f1*t))));        
    end

% Nested function representing the mass matrix of the system. When alpha0 =
% 0 the mass matrix is linear.
    function Mnonlin = Mnonlin_func(t,Z)
        
        alpha = zeros(1,N);
        Bnl=zeros(N,N);
        sat = 10^(-5); % parameter to be tuned to achive the saturation
        alpha0 = 0.75;
        for r=2:N-1
            dummy = 0;
            for s = 2:N-1
                
                
                % This is the version of alpha as suggested by Botti (2009)
                % 1)
                alphatemp =  (1/sqrt(lambda*pi))*(((1/sat)^2)*(Z(s*2,:))^2)...
                    * exp(-((r*delta-s*delta)^2));
                
                %Integration = area under function --> sum of function*delta
                dummy = dummy + alphatemp*delta;
                
            end
            
            % For big values of dummy tanh = 1 ---> alpha = 0
            alpha(r) = alpha0 * (1-tanh(dummy));
            
        end
        
        
        % Matrices for the nonlinear Mass function
        for o = 2:N-1-K
            Bnl(o+K,o)=-alpha(o);
        end
        
        Bnl(N,N)=0;
        Bnl = Bnl + eye(N);
        Cnl = -Bnl + eye(N);
        
        % Mass functions as suggested by Moleti(2009) - Feedforward
        % approximation
        
        G = inv(Bnl)*Cnl + eye(N);
        Mnonlin = I - ((Be*G)*inv(F))*Ce;
        Mnonlin=sparse(Mnonlin);
        
    end

% Saving the important parameters and matrices
save(sprintf('Casarini_C_DPOAEs_N%d_f0%d_a%d_r%3.2f.mat',N, f0,adB, roughness)...
    ,'Y', 'T', 'Index', 'position1', 'position2', 'position3', 'f0', 'f1', 'fdp');

% End of time measurement
toc;

end
