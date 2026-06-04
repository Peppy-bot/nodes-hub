// External KDL cross-check for the gravity / Coriolis modules.
//
// Unlike `ChainDynParam` (serial only), KDL's `TreeIdSolver_RNE` runs inverse
// dynamics over the whole *tree*, so it includes the parallel gripper whose two
// jaws branch off `link7`. With the fingers held at home (q = q_dot = 0), the
// torques it reports for the seven arm joints therefore carry the distal payload
// exactly as the Rust crate's gravity / Coriolis do (which lump that mass into
// the last segment). This is the reference used to regenerate the hard-coded
// `*_matches_kdl` arrays in gravity.rs / coriolis.rs on a Linux/ROS machine.
//
// Gravity is applied along world -Z and the tree roots at `openarm_body_link0`
// (== world), matching the crate's world-frame convention. The seven arm joints
// are read out by name (openarm_{side}_joint1..7), in JointVec order; the gripper
// joints are present in the tree but pinned to zero.
//
// Build (Orocos KDL + kdl_parser + urdfdom). With ROS or system packages:
//   g++ -std=c++17 -O2 tools/kdl_reference.cpp -o /tmp/kdl_reference \
//       $(pkg-config --cflags --libs orocos-kdl) -lkdl_parser -lurdfdom_model
// Or against a pixi/conda env that ships them (adjust ENV):
//   ENV=/path/to/env; g++ -std=c++17 -O2 tools/kdl_reference.cpp -o /tmp/kdl_reference \
//       -I$ENV/include -I$ENV/include/eigen3 -L$ENV/lib \
//       -lkdl_parser -lorocos-kdl -lurdfdom_model
// Run from the crate root (so the default URDF path resolves), or pass a path:
//   /tmp/kdl_reference [tests/fixtures/openarm_v10.urdf]
// A ROS 2 / ament build of kdl_parser also needs AMENT_PREFIX_PATH set, e.g.:
//   AMENT_PREFIX_PATH=$ENV LD_LIBRARY_PATH=$ENV/lib /tmp/kdl_reference
//
// On macOS (no ROS kdl_parser package): `brew install orocos-kdl urdfdom eigen`,
// then compile this alongside a standalone kdl_parser built by copying the
// `treeFromUrdfModel` conversion from ros/kdl_parser and parsing the URDF with
// urdfdom's `urdf::parseURDF` (the ROS `urdf` wrapper + rcutils are not needed).
// The reference arrays in gravity.rs / coriolis.rs were generated this way.

#include <kdl_parser/kdl_parser.hpp>
#include <kdl/jntarray.hpp>
#include <kdl/tree.hpp>
#include <kdl/treeidsolver_recursive_newton_euler.hpp>

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

constexpr unsigned int DOF = 7;
const double HALF_PI = std::acos(-1.0) / 2.0; // portable pi/2 (M_PI is non-standard)
using Vec7 = std::array<double, DOF>;

// Map every movable joint's name to its index in the tree's JntArray ordering.
std::map<std::string, unsigned int> joint_indices(const KDL::Tree& tree) {
    std::map<std::string, unsigned int> qnr;
    for (const auto& entry : tree.getSegments()) {
        const KDL::Segment& seg = GetTreeElementSegment(entry.second);
        const KDL::Joint& jnt = seg.getJoint();
        if (jnt.getType() != KDL::Joint::None) {
            qnr[jnt.getName()] = GetTreeElementQNr(entry.second);
        }
    }
    return qnr;
}

// Place a 7-vector onto the arm joints of `side`, leaving all other tree joints
// (the gripper) at their existing value.
void set_arm(KDL::JntArray& q, const std::map<std::string, unsigned int>& qnr,
            const std::string& side, const Vec7& v) {
    for (unsigned int i = 0; i < DOF; ++i) {
        q(qnr.at("openarm_" + side + "_joint" + std::to_string(i + 1))) = v[i];
    }
}

void print_arm(const std::string& label, const KDL::JntArray& tau,
               const std::map<std::string, unsigned int>& qnr, const std::string& side) {
    std::cout << "  " << std::left << std::setw(34) << label << "[";
    std::cout << std::fixed << std::setprecision(4);
    for (unsigned int i = 0; i < DOF; ++i) {
        double t = tau(qnr.at("openarm_" + side + "_joint" + std::to_string(i + 1)));
        // Match the test convention: values below the 1e-3 tolerance read as 0.
        if (std::abs(t) < 1e-3) t = 0.0;
        std::cout << (i ? ", " : "") << t;
    }
    std::cout << "]\n";
}

}  // namespace

int main(int argc, char** argv) {
    const std::string urdf = (argc > 1) ? argv[1] : "tests/fixtures/openarm_v10.urdf";

    KDL::Tree tree;
    if (!kdl_parser::treeFromFile(urdf, tree)) {
        std::cerr << "failed to parse URDF: " << urdf << "\n";
        return 1;
    }
    const auto qnr = joint_indices(tree);
    const unsigned int n = tree.getNrOfJoints();

    // Two solvers: gravity-only (q_dot = q_ddot = 0) and gravity-free (Coriolis
    // remains once q_ddot = 0).
    KDL::TreeIdSolver_RNE grav_solver(tree, KDL::Vector(0.0, 0.0, -9.81));
    KDL::TreeIdSolver_RNE cori_solver(tree, KDL::Vector(0.0, 0.0, 0.0));
    KDL::WrenchMap f_ext; // empty: no external forces

    // Postures / velocities must stay in sync with gravity.rs / coriolis.rs.
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
    // gravity, different chain, so their torques differ at the same posture.
    for (const std::string& side : {"left", "right"}) {
        std::cout << "=== " << side << " arm ===\n";

        std::cout << "JntToGravity (gravity = world -Z, gripper at home):\n";
        for (const auto& [label, q] : gravity_cases) {
            KDL::JntArray jq(n), zero(n), tau(n);
            set_arm(jq, qnr, side, q);
            grav_solver.CartToJnt(jq, zero, zero, f_ext, tau);
            print_arm(label, tau, qnr, side);
        }

        std::cout << "JntToCoriolis (gripper at home):\n";
        for (const auto& [label, q, qd] : coriolis_cases) {
            KDL::JntArray jq(n), jqd(n), zero(n), tau(n);
            set_arm(jq, qnr, side, q);
            set_arm(jqd, qnr, side, qd);
            cori_solver.CartToJnt(jq, jqd, zero, f_ext, tau);
            print_arm(label, tau, qnr, side);
        }
        std::cout << "\n";
    }

    return 0;
}
