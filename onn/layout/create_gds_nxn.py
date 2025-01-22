import gdsfactory as gf
import numpy as np
import re


def get_bottom_bs_indices(N):
    """
    Get indices of bottom beamsplitters in an NxN interferometer
    """
    if N % 2 != 0:
        raise ValueError("N must be even")

    bottom_indices = []

    # number of beamsplitters in the top triangle of the mesh
    num_top_tri_bs = np.array([2 * ii + 1 for ii in range(int(N / 2))])

    # start the first index by summing number of BS in the top triangle
    bottom_indices.append(np.sum(num_top_tri_bs[:-1]))

    # number of beamsplitters in the bottom triangle of the mesh
    num_bot_tri_bs = [2 * ii for ii in range(1, int(N / 2))]

    # Append the diagonal number of BS
    num_bot_tri_bs.append(num_top_tri_bs[-1])

    for ii in num_bot_tri_bs[-1:0:-1]:
        curr_idx = bottom_indices[-1]
        bottom_indices.append(curr_idx + ii)

    return np.int32(bottom_indices)


def match_beamsplitters(BS_list, N):
    """
    Matches beamsplitters by connecting their input and output ports based on mode numbers.

    Args:
        BS_list: List of beamsplitter objects, each with mode1 and mode2 attributes
        N: Integer representing the size of the interferometer

    Returns:
        tuple: (results, port_connections)
            - results: Dictionary mapping beamsplitter indices to their closest matches
            - port_connections: Dictionary mapping output ports to input ports
    """
    results = {}  # Store matching results for each beamsplitter
    port_connections = {}  # Store the final port-to-port connections
    bottom_bs = get_bottom_bs_indices(N)  # Get indices of bottom-row beamsplitters

    # Iterate through each beamsplitter to find its matches
    for i, bs1 in enumerate(BS_list):
        matches = []  # Store potential matches for current beamsplitter

        # Generate unique port names for current beamsplitter's outputs
        bs1_out_port1 = f"BS{i}_out{bs1.mode1}"
        bs1_out_port2 = f"BS{i}_out{bs1.mode2}"

        # Compare with all beamsplitters that come after the current one
        for j, bs2 in enumerate(BS_list):
            if j <= i:  # Skip previously processed beamsplitters
                continue

            # Generate unique port names for potential matching beamsplitter's inputs
            bs2_in_port1 = f"BS{j}_in{bs2.mode1}"
            bs2_in_port2 = f"BS{j}_in{bs2.mode2}"

            # Special handling for bottom-row beamsplitters
            if i in bottom_bs:
                # If both modes match (in any order), connect the higher mode number
                if (bs1.mode1 == bs2.mode1 and bs1.mode2 == bs2.mode2) or (
                    bs1.mode1 == bs2.mode2 and bs1.mode2 == bs2.mode1
                ):
                    # Select connection based on which mode number is higher
                    if bs1.mode1 > bs1.mode2:
                        port_match = (
                            bs1_out_port1,
                            bs2_in_port1 if bs1.mode1 == bs2.mode1 else bs2_in_port2,
                        )
                    else:
                        port_match = (
                            bs1_out_port2,
                            bs2_in_port1 if bs1.mode2 == bs2.mode1 else bs2_in_port2,
                        )
                # Check for single mode matches if both modes don't match
                elif bs1.mode1 == bs2.mode1:
                    port_match = (bs1_out_port1, bs2_in_port1)
                elif bs1.mode1 == bs2.mode2:
                    port_match = (bs1_out_port1, bs2_in_port2)
                elif bs1.mode2 == bs2.mode1:
                    port_match = (bs1_out_port2, bs2_in_port1)
                elif bs1.mode2 == bs2.mode2:
                    port_match = (bs1_out_port2, bs2_in_port2)
                else:
                    port_match = None
            else:
                # Regular matching for non-bottom beamsplitters
                # Connect ports when modes match between beamsplitters
                if bs1.mode1 == bs2.mode1:
                    port_match = (bs1_out_port1, bs2_in_port1)
                elif bs1.mode1 == bs2.mode2:
                    port_match = (bs1_out_port1, bs2_in_port2)
                elif bs1.mode2 == bs2.mode1:
                    port_match = (bs1_out_port2, bs2_in_port1)
                elif bs1.mode2 == bs2.mode2:
                    port_match = (bs1_out_port2, bs2_in_port2)
                else:
                    port_match = None

            # If a valid connection was found, add it to matches
            if port_match:
                matches.append((j, bs2, port_match))

        # Sort matches by how close they are to the current beamsplitter
        matches.sort(key=lambda x: abs(x[0] - i))

        # Keep only the two closest matches for each beamsplitter
        closest_matches = matches[:2] if len(matches) >= 2 else matches
        results[i] = closest_matches

        # Store the port connections for the closest matches
        for _, _, (out_port, in_port) in closest_matches:
            port_connections[out_port] = in_port

    return results, port_connections


def sort_ports(port_list, port_type="in"):
    # Define a function to extract the number after "in" from the port name
    def get_in_number(port):
        # Extract the number after "in"/'out' from port.name
        in_number = int(port.name.split(port_type)[1])
        return in_number

    # Sort the list using the custom key function
    sorted_ports = sorted(port_list, key=get_in_number)
    return sorted_ports


def remove_outputs(ports_list, bs_numbers, output_numbers):
    # Convert numbers to strings for pattern matching
    bs = "|".join(map(str, bs_numbers))
    outputs = "|".join(map(str, output_numbers))
    # Pattern matches BS{specified bs numbers}_out{specified output numbers}
    pattern = f"BS({bs})_out({outputs})"
    return [port for port in ports_list if not re.match(pattern, port.name)]


@gf.cell
def create_mesh_interferometer(
    N: int,
    tunable_BS,
    coupler_array,
    BS_list,
    spacing_x: float = 300,
    spacing_y: float = 100,
    draw_bbox: bool = False,
) -> gf.Component:
    """Creates an NxN interferometer mesh using tunable beamsplitters.

    Args:
        N: Number of input/output ports
        tunable_BS: Base tunable beamsplitter component
        spacing_x: Horizontal spacing between components
        spacing_y: Vertical spacing between components

    Returns:
        gf.Component: The complete mesh interferometer
    """
    # mesh = gf.Component(f"mesh_interferometer_{N}x{N}")
    mesh = gf.Component()

    # TODO: Change the couplers to function inputs, for portability (DONE)
    input_coupler_array = coupler_array(n=N, x_reflection=True)
    output_coupler_array = coupler_array(n=N, x_reflection=False)

    bs_sq_array = gf.Component()
    mode_tracker = np.zeros(N)
    bs_ref = [None for _ in BS_list]

    # Add everything to the mesh gf Component
    input_coupler_array_ref = mesh << input_coupler_array
    output_coupler_array_ref = mesh << output_coupler_array
    bs_sq_array_ref = mesh << bs_sq_array

    # Place beamsplitters in a triangular arrangement
    for ii, BS in enumerate(BS_list):
        # Create a reference to the tunable BS
        bs_ref[ii] = bs_sq_array << tunable_BS

        # Calculate position for this beamsplitter
        x = max(mode_tracker[BS.mode1 - 1], mode_tracker[BS.mode2 - 1])
        y = ((N - BS.mode1) + (N - BS.mode2)) / 2

        # Move beamsplitter to position
        bs_ref[ii].move((x * spacing_x, y * spacing_y))

        # Add ports with custom names
        bs_sq_array.add_port(f"BS{ii}_in{BS.mode2}", port=bs_ref[ii].ports["o1"])
        bs_sq_array.add_port(f"BS{ii}_in{BS.mode1}", port=bs_ref[ii].ports["o2"])
        bs_sq_array.add_port(f"BS{ii}_out{BS.mode1}", port=bs_ref[ii].ports["o3"])
        bs_sq_array.add_port(f"BS{ii}_out{BS.mode2}", port=bs_ref[ii].ports["o4"])

        # Update mode tracker
        mode_tracker[BS.mode1 - 1] = x + 1
        mode_tracker[BS.mode2 - 1] = x + 1

    # Finding the closest beamsplitters for all to connect to
    closest_matches, port_connections = match_beamsplitters(BS_list, N)

    # Connecting all beamsplitters
    for out_port, in_port in port_connections.items():
        # print(f"{out_port} -> {in_port}")
        route = gf.routing.route_single_sbend(
            bs_sq_array,
            port1=bs_sq_array.ports[out_port],
            port2=bs_sq_array.ports[in_port],
            cross_section=gf.cross_section.strip,
        )

    # Aligning the components so that they are evenly distributed
    bs_sq_array_ref.movex(input_coupler_array.size_info.width + spacing_x / 2)
    bs_sq_array_ref.movey((input_coupler_array_ref.ymax - bs_sq_array_ref.ymax) / 2)
    output_coupler_array_ref.movex(
        bs_sq_array_ref.xmax + spacing_x / 2 + output_coupler_array.size_info.width
    )

    # Draw bounding box to check alignment
    if draw_bbox:
        mesh << gf.components.bbox(bs_sq_array_ref, layer=(2, 0))
        mesh << gf.components.bbox(input_coupler_array_ref, layer=(2, 0))
        mesh << gf.components.bbox(output_coupler_array_ref, layer=(2, 0))

    # Connecting input and output
    mesh.add_ports(input_coupler_array_ref.ports, prefix="in")
    mesh.add_ports(output_coupler_array_ref.ports, prefix="out")
    mesh.add_ports(bs_sq_array_ref.ports)

    # Computing all the indices for the input side of mesh
    input_bs_idx = np.zeros(1)  # starts from 0 in the case of one BS
    num_top_tri_bs = np.array([2 * ii + 1 for ii in range(int(N / 2 - 1))])
    input_bs_idx = np.concatenate([input_bs_idx, np.cumsum(num_top_tri_bs)])

    # Output side of mesh
    num_bot_tri_bs = np.array([2 * ii for ii in range(int(N / 2 - 1))])
    output_bs_idx = N * (N - 1) / 2 - 1 - np.cumsum(num_bot_tri_bs)
    tmp1 = output_bs_idx[0]
    tmp2 = output_bs_idx[-1]
    output_bs_idx = np.concatenate(
        [
            [tmp1 - 1],
            output_bs_idx,
            [tmp2 - 2 * (N / 2 - 1)],
        ]
    )

    # Get names of beamsplitter
    in_pattern = f"BS({'|'.join(map(str, input_bs_idx.astype(np.int32)))})_in"
    input_ports_mesh = [
        port for port in mesh.get_ports_list() if re.match(in_pattern, port.name)
    ]

    out_pattern = f"BS({'|'.join(map(str, output_bs_idx.astype(np.int32)))})_out"
    output_ports_mesh = [
        port for port in mesh.get_ports_list() if re.match(out_pattern, port.name)
    ]

    # Get in/out couplers Port variable
    input_ports_couplers = [
        port for port in mesh.get_ports_list() if re.match(r"ino\d+", port.name)
    ]

    output_ports_couplers = [
        port for port in mesh.get_ports_list() if re.match(r"outo\d+", port.name)
    ]

    routes = gf.routing.route_bundle_sbend(
        mesh,
        ports1=input_ports_couplers[::-1],
        ports2=sort_ports(input_ports_mesh, port_type="in"),
        cross_section=gf.cross_section.strip,
        sort_ports=True,
    )

    sorted_output_mesh = sort_ports(output_ports_mesh, port_type="out")
    removed_output_mesh = remove_outputs(
        sorted_output_mesh,
        [output_bs_idx[0].astype(np.int32), output_bs_idx[-1].astype(np.int32)],
        [2, N - 1],  # always the second and second last mode
    )

    routes = gf.routing.route_bundle_sbend(
        mesh,
        ports1=removed_output_mesh,
        ports2=output_ports_couplers[::-1],
        cross_section=gf.cross_section.strip,
        sort_ports=True,
    )

    return mesh
