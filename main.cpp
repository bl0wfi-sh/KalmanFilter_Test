#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <chrono>
#include <eigen3/Eigen/Dense>
#include <math.h>

// Main function.
int main(int argc, char* argv[])
{
	std::string str;
	std::fstream f;
	f.open("/dev/ttyACM1");

	// Matrix's for kalman filter.
	Eigen::MatrixXf Q(2,2);				// Tunable parameters. To increase accuracy of observer. Process Noise. [sensor + model]
	Q << .01, 0,
		 0,   .01;

	Eigen::MatrixXf R(3,3);				// Comes from sensor noise. Just look at the data and see how much it wiggles. Variable Noise. [sensor + model]
	R << .001,   	0,		0,
		 0,    		.001,	0,
		 0,			0,		.001;

	Eigen::MatrixXf x_hat = Eigen::MatrixXf::Zero(2, 1);		// The thing we are estimating (System states) - Rad [pitch, roll]
	Eigen::MatrixXf P(2,2);			// Covariance of the estimation error.
	P << .01, 0, 0, .01;

	bool first = true;

	unsigned long long start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
	while (f >> str)
	{
		// Timing of the system.
		// Looks like without anything else we can run at max 200Hz.
		unsigned long long next = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
		unsigned long long elapsed = next - start;
		//std::cout << "Elapsed time! = " << elapsed << std::endl;
		start = next;

		// Remove SOP and EOP characters.
		str = str.substr(1, str.size() - 2);

		std::vector<std::string> data;
		std::stringstream ss(str);

		while( ss.good() )
		{
			std::string substr;
			getline(ss, substr, ',');
			data.push_back( substr );
		}

		float gx, gy, gz, ax, ay, az;				// rad/s
		gx = std::stof(data[0]) * M_PI / 180.0f;
		gy = std::stof(data[1]) * M_PI / 180.0f;
		gz = std::stof(data[2]) * M_PI / 180.0f;

		ax = std::stof(data[3]);
		ay = std::stof(data[4]);
		az = -1 * std::stof(data[5]);

		// Initialize kalman filter with first reading.
		if (first)
		{
			float accel_tot = sqrt( (ax*ax) + (ay*ay) + (az*az) );
			x_hat(0,0) = asin( ay / accel_tot );
			x_hat(1,0) = asin( ax / accel_tot );

			first = false;
			continue;
		}

		//Kalman Filter!
		/*** PREDICTION ***/

		// Propogating the states forward with X_hat = x_hat + f(x,u)
		Eigen::MatrixXf rot_matrix(2, 3);
		rot_matrix << 1, sin( x_hat(0,0) ) * tan( x_hat(1,0) ), cos( x_hat(0,0) ) * tan( x_hat(1,0)  ),
				 	  0, cos( x_hat(0,0) ),                     -sin( x_hat(0,0) );

		Eigen::MatrixXf gyros(3, 1);
		gyros << gx, gy, gz;

		Eigen::MatrixXf f_x_u = rot_matrix * gyros;

		x_hat = x_hat + (float)(elapsed / 1000.0f) * f_x_u;

		// Calculating A matrix.
		Eigen::MatrixXf A(2,2);
		A << gy * cos( x_hat(0,0) ) * tan( x_hat(1,0) ) - gz * sin( x_hat(0,0) ) * tan( x_hat(1,0) ), 	gy * sin( x_hat(0,0) ) * ( 1 / ( cos( x_hat(1,0) ) * cos( x_hat(1,0) ) )) + gz * cos( x_hat(0,0) ) * ( 1 / ( cos( x_hat(1,0) ) * cos( x_hat(1,0) ) )),
			 -gy * sin( x_hat(0,0) ) - gz * cos( x_hat(1,0) ),											0;

		Eigen::MatrixXf A_T(2,2);
		A_T = A.transpose();

		// Updating process covariance.
		P = P + (float)(elapsed / 1000.0f) * ( A * P + P * A_T + Q );

		/*** UPDATE ***/
		Eigen::MatrixXf h_x_u(3,1);
		h_x_u << 9.8 * sin( x_hat(1,0) ),
				-9.8 * cos( x_hat(1,0) ) * sin( x_hat(0,0) ),
				-9.8 * cos( x_hat(1,0) ) * cos( x_hat(0,0) );

		Eigen::MatrixXf accels(3,1);
		accels << ax,
				  ay,
				  az;

		Eigen::MatrixXf C(3,2);
		C << 0, 											9.8 * cos( x_hat(1,0) ),
			 -9.8 * cos( x_hat(1,0) ) * cos( x_hat(0,0) ),	9.8 * sin( x_hat(1,0) ) * sin( x_hat(0,0) ),
			 9.8 * cos( x_hat(1,0) ) * sin( x_hat(0,0) ),	9.8 * sin( x_hat(1,0) ) * cos( x_hat(0,0) ); 

		Eigen::MatrixXf C_T(2,3);
		C_T = C.transpose();

		Eigen::MatrixXf L(2,3);

		L = P * C_T * ((R + C*P*C_T).inverse());

		P = (Eigen::MatrixXf::Identity(2,2) - L*C)*P;

		x_hat = x_hat + L * (accels - h_x_u);
		
		std::cout << x_hat(0,0) * (180.0f / M_PI) << "," << x_hat(1,0) * (180.0f / M_PI) << std::endl;
	}

	std::cout << "Program ended, no new data in pipe." << std::endl;

}