# AII Flare prediction
### Authors: 
- Ildebrando Simeoni - ildebrando.simeoni@studio.unibo.it
- Davide Femia - davide.femia@studio.unibo.it
- Riccardo Falco - riccardo.falco2@tudio.unibo.it
- Vincenzo Collura - vincenzo.collura2@studio.unibo.it

## Local Execution (Preferred)

If you want to execute this project, you are strongly encouraged to run the following instructions:
* Install Docker, by following the [online instructions](https://docs.docker.com/get-docker/)
* Install Docker Compose, by following the [online instructions](https://docs.docker.com/compose/install/)
* Clone the repository
* Start the container via Docker Compose, from the main directory of the tutorial:

    ```
    docker-compose up
    ```

On linux systems, you may need to start the docker service first.

The process will end with a message such as this one:
```
To access the notebook, open this file in a browser:
    file:///home/lompa/.local/share/jupyter/runtime/nbserver-1-open.html
Or copy and paste this URL:
    http://127.0.0.1:39281/?token=0cd92163797c3b3abe67c2b0aea57939867477d6068708a2
```
Copying one of the two addresses in a file browser will provide access to the Jupyter server running in the spawned container. By default, the main lecture folders is shared with the container environment, so any modification you make in the contain will reflect in the host system, and the other way round.

Once you are done, pressing CTRL+C on the terminal will close the Docker container.

For more information about how Docker works (such as the difference between images and containers, or how to get rid of all of them once you are done with the tutorial), you can check the Docker documentation.

