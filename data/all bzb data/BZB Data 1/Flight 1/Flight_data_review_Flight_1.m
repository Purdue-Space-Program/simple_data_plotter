Data =  readmatrix('BZB Data_R-Spreadsheet.csv');

time = Data(:,1);

Altitude_GPS = Data(:,5);
Altitude_ACC = Data(:,4);
Figure_ALT_T = figure('Name','Altitude vs Time');
Plot_Alt_T = plot(time(),Altitude_GPS(),time(),Altitude_ACC);
title('Altitude vs Time');
xlabel('Time (seconds)');
ylabel('Altitude AGL (ft)');
legend('Altitude AGL - GPS','Altitude AGL - ACC');
xlim([0 50]);
grid on;
grapherPSP(Figure_ALT_T,Plot_Alt_T,'Light');


Acceleration = Data(:,2);
Figure_ALT_T = figure('Name','Acceleration vs Time');
Plot_Alt_T = plot(time(),Acceleration());
title('Acceleration vs Time');
xlabel('Time (seconds)');
ylabel("Acceleration G's");
xlim([0 50]);
grid on;
grapherPSP(Figure_ALT_T,Plot_Alt_T,'Light');



