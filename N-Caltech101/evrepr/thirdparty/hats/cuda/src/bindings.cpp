#include "group_by_cell.h"
#include "group_by_bin.h"
#include "n_events_cell.h"
#include "local_surface.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("group_by_cell", &group_by_cell, "Group by cell");
    m.def("group_by_bin", &group_by_bin, "Group by bin");
    m.def("n_events_cell", &n_events_cell, "Num events in each cell");
    m.def("local_surface", &local_surface, "Local time surface");
}
