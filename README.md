# 2022_04_13_ray_serve_meetup_demo

Code samples for Ray Serve Meetup on 04/13/2022

- Download image caption pre-trained model by running `make prep`
- `python deployment_graph.py` should execute the dag locally
- `serve run deployment_graph.serve_entrypoint` to deploy to Serve cluster.
