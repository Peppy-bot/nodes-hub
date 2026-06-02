// Reference generator for the gravity / Coriolis values asserted in
// `src/dynamics/gravity.rs` and `src/dynamics/coriolis.rs`.
//
// It loads the same URDF and chain the Rust crate uses and prints KDL's
// `ChainDynParam::JntToGravity` / `JntToCoriolis` for the same (q, q_dot)
// postures, so the hand-recorded test arrays can be regenerated and re-checked
// whenever the URDF inertials change. The Rust RNEA must match this output to
// the 1e-3 N*m tolerance the tests use.
//
// Chain + gravity: the crate's dynamics run in the WORLD frame (the arm base is
// mounted with rpy = (-pi/2, 0, 0)), so the reference roots at the torso base
// `openarm_body_link0` (== world) and applies gravity along world -Z. The seven
// movable joints of the chain are j1..j7, matching the Rust `JointVec` order.
//
// Build (Orocos KDL + kdl_parser + urdfdom). With ROS or system packages:
//   g++ -std=c++17 -O2 tools/kdl_reference.cpp -o /tmp/kdl_reference \
//       $(pkg-config --cflags --libs orocos-kdl) -lkdl_parser -lurdfdom_model
// Or against a pixi/conda env that ships them (adjust ENV):
//   ENV=/path/to/env; g++ -std=c++17 -O2 tools/kdl_reference.cpp -o /tmp/kdl_reference \
//       -I$ENV/include -I$ENV/include/eigen3 -L$ENV/lib \
//       -lkdl_parser -lorocos-kdl -lurdfdom_model
// Run from the crate root (so the default URDF path resolves), or pass a path:
//   /tmp/kdl_reference [urdf/openarm_v10.urdf]
// A ROS 2 / ament build of kdl_parser also needs AMENT_PREFIX_PATH set, e.g.:
//   AMENT_PREFIX_PATH=$ENV LD_LIBRARY_PATH=$ENV/lib /tmp/kdl_reference
//
// Verified: this reproduces every value in gravity.rs / coriolis.rs to 4 dp.

#include <kdl_parser/kdl_parser.hpp>
#include <kdl/chain.hpp>
#include <kdl/chaindynparam.hpp>
#include <kdl/jntarray.hpp>
#include <kdl/tree.hpp>

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

constexpr unsigned int DOF = 7;
const double HALF_PI = std::acos(-1.0) / 2.0; // portable pi/2 (M_PI is non-standard)
using Vec7 = std::array<double, DOF>;

KDL::JntArray to_jnt(const Vec7& v) {
    KDL::JntArray j(DOF);
    for (unsigned int i = 0; i < DOF; ++i) j(i) = v[i];
    return j;
}

void print_row(const std::string& label, const KDL::JntArray& tau) {
    std::cout << "  " << std::left << std::setw(34) << label << "[";
    std::cout << std::fixed << std::setprecision(4);
    for (unsigned int i = 0; i < DOF; ++i) {
        // Match the test convention: values below the 1e-3 tolerance read as 0.
        double v = (std::abs(tau(i)) < 1e-3) ? 0.0 : tau(i);
        std::cout << (i ? ", " : "") << v;
    }
    std::cout << "]\n";
}

}  // namespace

int main(int argc, char** argv) {
    const std::string urdf = (argc > 1) ? argv[1] : "urdf/openarm_v10.urdf";

    KDL::Tree tree;
    if (!kdl_parser::treeFromFile(urdf, tree)) {
        std::cerr << "failed to parse URDF: " << urdf << "\n";
        return 1;
    }

    // Postures must stay in sync with the gravity.rs / coriolis.rs tests.
    const std::vector<std::pair<std::string, Vec7>> gravity_cases = {
        {"home", {0, 0, 0, 0, 0, 0, 0}},
        {"q1 = pi/2", {HALF_PI, 0, 0, 0, 0, 0, 0}},
        {"q4 = pi/2", {0, 0, 0, HALF_PI, 0, 0, 0}},
        {"mixed", {0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7}},
    };
    const std::vector<std::tuple<std::string, Vec7, Vec7>> coriolis_cases = {
        {"q=0, qd1=5", {0, 0, 0, 0, 0, 0, 0}, {5, 0, 0, 0, 0, 0, 0}},
        {"q=0, qd4=5", {0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 5, 0, 0, 0}},
        {"q4=pi/2, qd=(3,_,_,3)", {0, 0, 0, HALF_PI, 0, 0, 0}, {3, 0, 0, 3, 0, 0, 0}},
        {"mixed q, mixed qd",
         {0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7},
         {1.0, -1.5, 2.0, -2.5, 3.0, -3.5, 4.0}},
    };

    // The left and right arms are mirror images: same body root and world -Z
    // gravity, different tip chain, so their torques differ at the same posture.
    for (const std::string& side : {"left", "right"}) {
        const std::string tip = "openarm_" + side + "_link7";
        KDL::Chain chain;
        if (!tree.getChain("openarm_body_link0", tip, chain)) {
            std::cerr << "failed to extract chain openarm_body_link0 -> " << tip << "\n";
            return 1;
        }
        if (chain.getNrOfJoints() != DOF) {
            std::cerr << "expected " << DOF << " joints, chain has " << chain.getNrOfJoints() << "\n";
            return 1;
        }
        KDL::ChainDynParam dyn(chain, KDL::Vector(0.0, 0.0, -9.81));

        std::cout << "=== " << side << " arm ===\n";
        std::cout << "JntToGravity (gravity = world -Z):\n";
        for (const auto& [label, q] : gravity_cases) {
            KDL::JntArray jq = to_jnt(q), tau(DOF);
            dyn.JntToGravity(jq, tau);
            print_row(label, tau);
        }
        std::cout << "JntToCoriolis:\n";
        for (const auto& [label, q, qd] : coriolis_cases) {
            KDL::JntArray jq = to_jnt(q), jqd = to_jnt(qd), tau(DOF);
            dyn.JntToCoriolis(jq, jqd, tau);
            print_row(label, tau);
        }
        std::cout << "\n";
    }

    return 0;
}
