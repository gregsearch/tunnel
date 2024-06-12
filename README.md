# Orientation

<img width="337" alt="image" src="https://github.com/gregsearch/tunnel/assets/20931037/f98685d6-e7c8-466f-a3ef-612f02ad8b13">

The task is to fly through the Tunnel without impacting the sides. The main section of the display is a rear view of the aircraft. The right side of the display is a top-down view which is a scaled representation of the depth of the Tunnel.

# Observation Space

Three can be used as part of the observation space. The first is NX. This refers to information about the aircrafts' state. The elements of NX are defined as in the workd by Oswin So AND ChuChu Fan in: https://arxiv.org/pdf/2305.14154

<img width="600" alt="image" src="https://github.com/gregsearch/tunnel/assets/20931037/db26fe31-156a-40e5-b2ac-95ae5ecef574">

The second element available for the observation space is the "echomap" which refers to a distance measurement to the wall in the direction of the sensor nodes. The sensor nodes are depicted as green lines and are fixed relative to the aircraft orientation as it rolls, pitches and yaws.

Third is the targets. 

# Action Space 

The actions are labelled NU. It is a four element array [Nz, Ps, Nyr, Throttle]. An increase in each element corresponds to aft control stick, right control stick, right rudder and increase throttle, respectively. 
