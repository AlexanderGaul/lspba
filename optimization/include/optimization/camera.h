#ifndef CAMERA_H
#define CAMERA_H

#include <ceres/ceres.h>


template <typename S>
class RadialCamera {
  public:
    static constexpr int N = 6;
    typedef Eigen::Matrix<S, 2, 1> Vec2;
    typedef Eigen::Matrix<S, 3, 1> Vec3;
    typedef Eigen::Matrix<S, N, 1> VecN;
    
    RadialCamera(const VecN& p) : param(p) {};
    
    /* TODO is it allowed to write over same varaible??? */
    /* maybe check this in the basalt headers?? */
    Vec2 project(const Vec3& p, 
                 Eigen::Matrix<S, 2, 3>* jacobian=nullptr,
                 Eigen::Matrix<S, 2, N>* jacobian_param=nullptr) const {
        Vec2 res;
        Vec2 x_proj;
        Vec2 x_distort;
        // TODO: name intermediate results
        x_proj[0] = p[0] / p[2];
        x_proj[1] = p[1] / p[2];

        S r = x_proj.norm();
        S distortion = 1. + r * r * (param[4] + r * r * (param[5]));
        x_distort[0] = distortion * x_proj[0];
        x_distort[1] = distortion * x_proj[1];

        res[0] = param[0] * x_distort[0] + param[2];
        res[1] = param[1] * x_distort[1] + param[3];
        
        // TODD: calculate derivative
        if (jacobian) {
            Eigen::Matrix<S, 2, 3> jac_perspective;
            jac_perspective << S(1) / p[2], S(0), -p[0]/(p[2]*p[2]),
                               S(0), S(1) / p[2], -p[1]/(p[2]*p[2]);
            Eigen::Matrix<S, 2, 2> jac_distortion = 
                Eigen::Matrix<S, 2, 2>::Identity() * distortion + 
                x_proj * x_proj.transpose() * (S(2.) * param[4] + S(4.) * param[5] * r*r);
            Eigen::Matrix<S, 2, 2> jac_calibration;
            jac_calibration << param[0], S(0), S(0), param[1];
            Eigen::Matrix<S, 2, 3> jac = jac_calibration * jac_distortion * jac_perspective;
            *jacobian = jac;
        }
        if (jacobian_param) {
            jacobian_param->col(0) = Vec2(x_distort[0], 0);
            jacobian_param->col(1) = Vec2(0, x_distort[1]);
            jacobian_param->col(2) = Vec2(1, 0);
            jacobian_param->col(3) = Vec2(0, 1);
            jacobian_param->col(4) = r * r * (x_proj.array() * param.template head<2>().array());
            jacobian_param->col(5) = r * r * r * r * (x_proj.array() * param.template head<2>().array());
        }
        return res;
    }
    
    Vec3 unproject(const Vec2& p,
                   Eigen::Matrix<S, 3, N>* jacobian_param=nullptr) const {
        Vec3 res;
        Vec2 uncalibrated; /* TODO: rename */
        uncalibrated[0] = (p[0] - param[2]) / param[0];
        uncalibrated[1] = (p[1] - param[3]) / param[1];
        
        S b1 = - param[4];
        S b2 = 3. * param[4] * param[4] - param[5];
        S b3 = - 12. * param[4] * param[4] * param[4] + 
                    8. * param[4] * param[5];
        S b4 = 55. * param[4] * param[4] * param[4] * param[4] - 
                    55. * param[4] * param[4] * param[5] + 
                    5. * param[5] * param[5];
        S b5 = -273. * param[4] * param[4] * param[4] * param[4] * param[4] + 
                    364. * param[4] * param[4] * param[4] * param[5] - 
                    78. * param[4] * param[5] * param[5];
        S b6 = 1428. * param[4] * param[4] * param[4] * param[4] * param[4] * param[4] - 
                    2380. * param[4] * param[4] * param[4] * param[4] * param[5] + 
                    840. * param[4] * param[4] * param[5] * param[5] - 
                    35. * param[5] * param[5] * param[5];
                
        Eigen::Matrix<S, 6, 1> bs;
        bs << b1, b2, b3, b4, b5, b6;
        
        S r = uncalibrated.norm();
        S r2 = r * r;
        S r4 = r2 * r2;
        S r6 = r4 * r2;
        S r8 = r6 * r2;
        S r10 = r8 * r2;
        S r12 = r10 * r2;
        Eigen::Matrix<S, 6, 1> r_powers;
        r_powers << r2, r4, r6, r8, r10, r12;
        
        S Q = S(1.) + S(bs.transpose() * r_powers);
        
        res[0] = uncalibrated[0] * Q;
        res[1] = uncalibrated[1] * Q;
        res[2] = S(1.);
        
        if  (jacobian_param) {
            Eigen::Matrix<S, 2, 4> duncal_dparam;
            duncal_dparam << -(p[0] - param[2]) / (param[0] * param[0]), S(0.), S(-1.) / param[0], S(0.),
                             S(0.), -(p[1] - param[3]) / (param[1] * param[1]), S(0.), S(-1.) / param[1];
            Eigen::Matrix<S, 1, 2> dr2_duncal;
                dr2_duncal << S(2.) * uncalibrated[0], S(2.) * uncalibrated[1];
            Eigen::Matrix<S, 6, 1> drpowers_dr2;
            drpowers_dr2 << S(1), S(2.)*r2, S(3.)*r4, S(4.)*r6, S(5.)*r8, S(6.)*r10;
            Eigen::Matrix<S, 1, 1> dQ_dr2 = bs.transpose() * drpowers_dr2;
            Eigen::Matrix<S, 3, 1> dres_dQ{uncalibrated[0], uncalibrated[1], S(0.)};
            Eigen::Matrix<S, 3, 2> dres_duncal;
            dres_duncal << Q, S(0.),
                           S(0.), Q, 
                           S(0.), S(0.);

            jacobian_param->template block<3, 4>(0, 0) = dres_dQ * dQ_dr2 * dr2_duncal * duncal_dparam +
                                                         dres_duncal * duncal_dparam;
            
            Eigen::Matrix<S, 6, 2> db_dparam;
            db_dparam << S(-1.), S(0.),
                         S(3.*2.)*param[4], 
                                 S(-1.),
                         S(-12.*3.)*param[4] * param[4] + S(8.)*param[5], 
                                 S(8.)*param[4],
                         S(55.*4.)*param[4]*param[4]*param[4] - S(55.*2)*param[4]*param[5], 
                                 S(-55.)*param[4]*param[4] + S(5.*2.)*param[5], 
                         S(-273.*5.)*pow(param[4], 4.) + S(364.*3.)*param[4]*param[4]*param[5] - S(78.)*param[5]*param[5], 
                                 S(364.)*param[4]*param[4]*param[4] - S(78.*2.)*param[4]*param[5], 
                         S(1428.*6.)*pow(param[4], 5.) - 
                         S(2380.*4.)*param[4]*param[4]*param[4]*param[5] + 
                         S(840.*2.)*param[4]*param[5]*param[5], 
                                 S(-2380.)*param[4]*param[4]*param[4]*param[4] + 
                                 S(840.*2.)*param[4]*param[4]*param[5] -
                                 S(35.*3.)*param[5]*param[5];
            Eigen::Matrix<S, 1, 6> dQ_db = r_powers.transpose();
            
            jacobian_param->template block<3, 2>(0, 4) = dres_dQ * dQ_db * db_dparam;
        }
        
        return res;
    }
    
    
    
  private:
    VecN param;
};


#endif // CAMERA_H
