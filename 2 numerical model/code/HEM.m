clear;
clc

W = [];  % save mass flow rate results

data = csvread('experimental data D1.csv', 1, 0);  % load input data
[num, ~] = size(data);  % the number of input data

for n = 1 : num
n

% input parameters
ldr = data(n, 1);  % length to diameter ratio
pre = data(n, 2);  % pressure (MPa)
dia = data(n, 3);  % diameter (mm)
miu = data(n, 4);  % roughness (μm)
tem = data(n, 5);  % temperature (K)

L = ldr * dia;  % length of pipe (mm)
A = 0.25 * pi * (0.001 * dia) ^ 2;  % area (m^2)

% set iteration parameters
pc0 = pre - 0.05;  % initial guess of output pressure (MPa)
dpc = -0.002;  % step size of output pressure (MPa)
epsilon = 0.001;  % accuracy (MPa)
C = 0.84;  % discharge coefficient

% calculate initial properties
h0 = 0.001 * refpropm('H', 'T', tem, 'P', 1000 * pre, 'CO2');  % enthalpy (kJ/kg)
s0 = 0.001 * refpropm('S', 'T', tem, 'P', 1000 * pre, 'CO2');  % Entropy [kJ/(kg K)]
d0 = refpropm('D', 'T', tem, 'P', 1000 * pre, 'CO2');  % density (kg/m^3)
u0 = refpropm('V', 'T', tem,'P', 1000 * pre, 'CO2');  % dynamic viscosity (Pa*s)

% iteration
pc = pc0;
pc_p = pc + 3 * dpc; 
while(abs(pc - pc_p) >= epsilon )
    pc = pc + dpc;

    % two-phase region
    if(pc < 7.377)
        sl = 0.001 * refpropm('S', 'P', 1000 * pc, 'Q', 0, 'CO2');
        sg = 0.001 * refpropm('S', 'P', 1000 * pc, 'Q', 1, 'CO2');
        hl = 0.001 * refpropm('H', 'P', 1000 * pc, 'Q', 0, 'CO2');
        hg = 0.001 * refpropm('H', 'P', 1000 * pc, 'Q', 1, 'CO2');
        dl = refpropm('D', 'P', 1000 * pc, 'Q', 0, 'CO2');
        dg = refpropm('D', 'P', 1000 * pc, 'Q', 1, 'CO2');
        ul = refpropm('V', 'P', 1000 * pc, 'Q', 0, 'CO2');
        ug = refpropm('V', 'P', 1000 * pc, 'Q', 1, 'CO2');
        x = (s0 - sl) / (sg - sl);  % mass quality
        if (x > 0 && x < 1)
            hc = x * hg + (1 - x) * hl;
            vc = x / dg + (1 - x) / dl;
            dc = 1 / vc;
            uc = (x / ug + (1 - x) / ul) ^ (-1);
        end
    else
        hc = 0.001 * refpropm('H', 'P', 1000 * pc, 'S', 1000 * s0, 'CO2');
        dc = refpropm('D', 'P', 1000 * pc, 'S', 1000 * s0, 'CO2');
        uc = refpropm('V', 'P', 1000 * pc, 'S', 1000 * s0, 'CO2');
    end
    w = sqrt(2 * (h0 - hc) * 1000);  % velocity (m/s)
    G = w * dc;  % mass flow [kg/(m^2 s)]

    % Define the Colebrook equation as a function to solve friction coefficient f
    Re = G * dia * 0.001 / ((uc + u0) / 2);  % Reynolds number
    colebrook = @(f) 1/sqrt(f) + 2*log10((miu * 0.001 / 3.7 / dia) + (2.51 / Re / sqrt(f)));  % the Colebrook equation
    options = optimset('Display', 'off');  % suppress output
    f = fsolve(colebrook, 0.02, options);

    dp_c = G ^ 2 / d0 / (2 * C ^ 2) / 1e6;  % form drag pressure drop (MPa)
	dp_a = (G ^ 2 * (1 / dc - 1 / d0)) / 1e6;  % acceleration pressure drop (MPa)
    dp_f = f / 2 * L / dia * G ^ 2 * 2 / (dc + d0)  / 1e6;  %frictional pressure drop (MPa)
	dp = dp_c+ dp_a + dp_f;  % total pressure drop (MPa)
	pc_p = pre - dp;
    W0 = 1000 * A * G;  % mass flow [g/(s)]
    if(pc < pc_p)
        break
    end
end
W(n, :) = W0;  % save results
W
end
csvwrite('HEM_D1.csv', W);