
### Search the registry for an image.
### Example: 'docker.io/library/imagename' or 'busybox'
### By default all entires in unqualified of 'registries.conf' are used.
### https://www.youtube.com/watch?v=YXfA5O5Mr18
$ podman search <image_name>
$ podman search busybox.
$ podman pull <image_name>
$ podman pull docker.io/library/busybox
$ podman images 
$ docker.io/library/busybox  latest      f62daa0d2c72  11 months ago  4.5 MB
$ podman rmi f62daa0d2c72
$ podman run -it <image_name>
$ podman run -it --rm <image_name>
$ podman ps 
$ podman ps -a
$ podman run --name <container_name> -p ext_port:int_port <container_image> 
$ podman start <container_name>
$ podman inspect <container_name>
$ podman search nginx
$ podman run --name pdm-nginx -p 8080:80 nginx
$ podman inspect <container_id>
$ podman port <container_id>
$ curl localhost:8080
$ podman stop <container_id>
$ podman build -t <image_name>
$ podman run --name <container_name> -p 8080:8080 <image_name>:<tag>
$ podman build -t <name>
$ podman login <registry-name>
$ podman build -t <username>/<image_run>
$ podman push <username>/<container_name>
$ podman pod --help
$ podman pod create --name <pod_name>
$ podman pod ls
$ podman ps -a --pod
$ podman pod create --name my_pod
### pod creates a default container in the pd.
$ podman ps -a --pod
$ podman run -dt --pod <pod_name> <container_image_name>
$ podman pod start <pod_name>
$ podman pod stop <pod_name>
$ podman pod rm <pod_name>
$ podman run -it --name=test1 docker.io/alpine sh
$ podman run -it --name=test2 docker.io/alpine sh
$ podman top test1
$ podman exec -it test1 sh
$ pomman exec test1 ifconfig tap0
$ uname -a
$ podman run -it --name=test1 --hostname=test1.example.com docker.io/alpine sh
$ podman run -it --name=test2 --user 12345 docker.io/alpine sh
$ cat /etc/passwd
$ ps -ef | grep 12345
$ free -h
$ podman stats test1
$ podman run -it --name=test1 --memory=50M -d docker.io/alpine sh
$ ls /etc/containers/registries.conf
$ grep ^registries /etc/containers/registries.conf
$ ['registry.access.redhat.com', 'registry.redhat.io', 'docker.io']
$ podman run -it --name=test1 docker.io/mysql
$ ping https://podman.io/docs/installation
$ podman run -it docker.io/alpine echo hello
$ podman pull docker.io/alpine
$ podman run docker.io/library/alpine:latest echo Hellooo World
$ podman run --help
$ podman search ubuntu --limit 3
$ podman pull docker.io/library/ubuntu
$ podman run docker.io/library/ubuntu echo hiiii
$ podman run --name=test-con1 docker.io/ubuntu bash
$ podman run --interactive --tty --name=test-con1 docker.io/ubuntu bash
$ hostname
$ cat /etc/lsb-release
$ uname -a
$ podman start test-con1
$ podman exec test-con1 hostname
$ podman stop|kill test-con1
$ podman kill test-con1
$ podman rm test-con1
$ podman run -it -d docker.io/library/ubuntu bash
$ podman exec test-con1 ps -elf
$ podman search httpd --limit 3
$ podman pull docker.io/library/httpd
$ podman inspect --help
$ podman inspect docker.io/library/httpd:latest | more     // look for 'ExposedPorts' == 80/tcp
$ podman run -d -p 8080:80 docker.io/library/httpd:latest  // run container in detatched mode and ind 8080 of host to 80 of container.
$ podman ps -a
$ CONTAINER ID  IMAGE                           COMMAND           CREATED         STATUS             PORTS                 NAMES
$ ff113802be24  docker.io/library/httpd:latest  httpd-foreground  39 seconds ago  Up 39 seconds ago  0.0.0.0:8080->80/tcp  lucid_bouman
$ curl http://localhost:8080
$ podman stop ff113802be24
$ podman start ff113802be24
$ podman exec -it ff113802be24 bash
$ podman cp --help
$ podman cp ff113802be24:/usr/local/apache2/htdocs/index.html .
$ podman cp index.html ff113802be24:/usr/local/apache2/htdocs/index.html
$ podman run --help

Note:
rootfull vs rootless containers
One of the guiding factors of netowrking for containers with Podman is going to be 
weather or not a container is run by a root, user or not.  This is because unpriviliged
users can not create networking interfaces on the host.  Therefore, for rootless containers,
the default netowork mode is slirp4netns.  Because of the limmited privileges, slirp4netns lacks
some of the features of netowrking compared to rootful Podman's networkng; for example, slirp4netns
can ot give containers a routable IP address, The default networking mode for rootful containers on the other side is netavark, which
allows container to have a routable IP address.

$ sudo -i
$ podman search mysql-57 --limit 3
$ podman pull docker.io/enekui/mysql-57-rhel7
$ podman inspect docker.io/enekui/mysql-57-rhel7:latest | more
$ "ExposedPorts": {
$     "3306/tcp": {}
$ podman run docker.io/enekui/mysql-57-rhel7
$ podman run --name=friends-db -p 3306:3306 -e MYSQL_USER=user1 -e MYSQL_PASSWORD=pass1234 -e MYSQL_DATABASE=friends -e MYSQL_ROOT_PASSWORD=rootpass -d  docker.io/enekui/mysql-57-rhel7
$ podman ps
$ CONTAINER ID  IMAGE                                   COMMAND     CREATED         STATUS             PORTS                   NAMES
$ 0b55e7d5086d  docker.io/enekui/mysql-57-rhel7:latest  run-mysqld  17 seconds ago  Up 17 seconds ago  0.0.0.0:3306->3306/tcp  friends-db
$ podman logs friends-db
$ podman inspect friends-db | more
$ "NetworkSettings": {
$            "EndpointID": "",
$            "Gateway": "10.88.0.1",
$            "IPAddress": "10.88.0.3",
$ podman search apache-php --limit 3
$ podman pull docker.io/newdeveloper/apache-php
$ podman inspect docker.io/newdeveloper/apache-php | more
$         "Config": {
$            "ExposedPorts": {
$                "80/tcp": {}
$
$ podman run --name=friends-apps -p 8080:80 -e MYSQL_USER=user1 -e MYSQL_PASSWORD=pass1234 -e MYSQL_DATABASE=friends -e DBHOST=mysql:10.88.0.3 -d docker.io/newdeveloper/apache-php

Note:
use git to down load some client/server application 

$ podman ps -a
$ CONTAINER ID  IMAGE                                     COMMAND            CREATED         STATUS                     PORTS                   NAMES
$ 04f3abc04cef  docker.io/enekui/mysql-57-rhel7:latest    run-mysqld         26 minutes ago  Exited (1) 26 minutes ago                          cranky_khayyam
$ 0b55e7d5086d  docker.io/enekui/mysql-57-rhel7:latest    run-mysqld         22 minutes ago  Up 22 minutes ago          0.0.0.0:3306->3306/tcp  friends-db
$ 0a6c3dfb14de  docker.io/newdeveloper/apache-php:latest  /bin/bash /run.sh  5 minutes ago   Up 5 minutes ago           0.0.0.0:8080->80/tcp    friends-apps

$ podman cp <client-server> 0a6c3dfb14de:<some-location>/index.php
$ curl http://localhost:8080/index.php
$ podman exec -it friends-db bash  // add records to the db in friends-db container.
$ curl http://10.88.0.4:8080
$ podman run -d -it docker.io/library/alpine sh
$ podman ps -a
$ CONTAINER ID  IMAGE                            COMMAND     CREATED             STATUS                 PORTS       NAMES
$ 82117644a47e  docker.io/library/alpine:latest  sh          About a minute ago  Up About a minute ago              elated_lalande
$ podman exec -it elated_lalande sh
$ podman ps -s
$ podman diff --help
$ podman diff elated_lalande
$ podman search alpine --limit 3
$ podman search docker.io/library/alpine --list-tags | more
$ grep search /etc/containers/
$ podman system df
$ podman system prune
$ podman diff alpinebash
$ podman ps -a 
$ CONTAINER ID  IMAGE                            COMMAND     CREATED        STATUS                     PORTS       NAMES
$ a0779e9ea2db  docker.io/library/alpine:latest  sh          2 minutes ago  Exited (0) 50 seconds ago              alpinebash
$ podman commit alpinebash alpinebash2
$ podman images
$ REPOSITORY                TAG         IMAGE ID      CREATED        SIZE
$ localhost/alpinebash2     latest      99dd10114b24  4 seconds ago  12 MB
$ docker.io/library/alpine  latest      05455a08881e  3 months ago   7.67 MB
$ podman history 99dd10114b24
$ podman build -f Containerfile
$ podman build -f Containerfile -t myhttpd3
$ podman history myhttpd3
$ podman inspect myhttpd3 | more
$ podman run -it --rm myhttpd3 bash 
$ podman inspect myhttpd3 | more
$            "Cmd": [
$                "/bin/sh",
$                "-c",
$                "/sbin/httxt2dbm -DFOREGROUND"
$            "Cmd": [
$                "/sbin/httxt2dbm",
$                "-DFOREGROUND"
$ podman run -p 8080:80 -d myhttpd3
$ curl http://localhost:8080
$ touch test.html
$ touch abc.html
$ tar czvf src.tar.gz *.html

Note:
The Containerfile has EXPOSE 80 added to the instructions.
$ podman run -it -d -p 8080:80 myhttpd3:latest bash
$ podman port -l
$ 80/tcp -> 0.0.0.0:8080
$ podman run -P -d myhttpd3
$ podman run -it -P -d myhttpd3 bash
$ podman port -l
$ 80/tcp -> 0.0.0.0:37249
$ 8080/tcp -> 0.0.0.0:35843
$ podman inpsect myhttpd3
$         "Config": {
$            "ExposedPorts": {
$                "80/tcp": {},
$                "8080/tcp": {}
$            },
$            "Env": [
$                "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
$            ],
$            "Cmd": [
$                "/sbin/httxt2dbm",
$                "-DFOREGROUND"
$            ],
$            "Volumes": {
$                "/data": {}
$ podman run -it -rm -e APP_APIKEY=BEEFDEAD myhttpd3 bash
$ podman build -f Containerfile4 -t simple_http_server
$ podman run -it -P localhost/simple_http_server bash
$ curl localhost:8080
$ podman ps -a
$ CONTAINER ID  IMAGE                                COMMAND               CREATED             STATUS                 PORTS                  NAMES
$ 04a596c7a249  localhost/simple_http_server:latest  /bin/sh -c python...  About a minute ago  Up About a minute ago  0.0.0.0:38201->80/tcp  vibrant_ptolemy
$ podman top vibrant_ptolemy
$ USER        PID         PPID        %CPU        ELAPSED          TTY         TIME        COMMAND
$ root        1           0           0.000       1m16.95187492s   ?           0s          /bin/sh -c python3 -m http.server 80 
$ root        2           1           0.000       1m16.951971923s  ?           0s          python3 -m http.server 80 
$ ps -ef | grep http.server
$ monteca+ 1688680 1688677  0 08:45 ?        00:00:00 /bin/sh -c python3 -m http.server 80
$ monteca+ 1688682 1688680  0 08:45 ?        00:00:00 python3 -m http.server 80
$ monteca+ 1688920 1684452  0 08:49 pts/0    00:00:00 grep --color=auto http.server
$ podman port -l
$ podman build -f Containerfile6 -t test
$ podman run --rm test
$ podman run --rm test ping -c5 www.google.com
$ podman run --rm test www.google.com
$ podman system prune
$ podman run -it localhost/myansi_capp:latest
$ podman history localhost/myansi_capp:latest

Note: 
	https://hub.docker.com/
	https:/quay.io                               // login with user id aradf
	https://cloud.google.com/artifact-registry/
	https://quay.io/repository/?tab=settings

$ podman images
$ REPOSITORY                TAG         IMAGE ID      CREATED        SIZE
$ localhost/myansic_app     latest      f4af75fab97d  3 minutes ago  78.8 MB
$ docker.io/library/ubuntu  latest      bf3dc08bfed0  13 days ago    78.7 MB
$ podman tag localhost/myansic_app:latest quay.io/aradf/myansi_capp:ubuntu24
$ podman images
$ REPOSITORY                 TAG         IMAGE ID      CREATED         SIZE
$ localhost/myansic_app      latest      f4af75fab97d  19 minutes ago  78.8 MB
$ quay.io/aradf/myansi_capp  ubuntu24    f4af75fab97d  19 minutes ago  78.8 MB
$ quay.io/aradf/myansi_capp  latest      f4af75fab97d  19 minutes ago  78.8 MB
$ docker.io/library/ubuntu   latest      bf3dc08bfed0  13 days ago     78.7 MB
$
$ podman login quay.io
$ aradf
$ ********
$ podman push quay.io/aradf/myansi_capp:ubuntu24
$ podman push quay.io/aradf/myansi_capp:latest
$ podman save quay.io/aradf/myansi_capp:latest > my_tarfile.tar
$ podman load -i my_tarfile.tar
$ sudo podman info | grep -A3 network
$ network:
$  - bridge
$  - macvlan
$  volume:
$ sudo podman network ls
$ NETWORK ID    NAME        VERSION     PLUGINS
$ 2f259bab93aa  podman      0.4.0       bridge,portmap,firewall,tuning

Note: 
	bridge is the default Driver for Podman and Docker.
	
$ sudo podman run -it alpine sh         // create a root full container, since sudo is used.
/# ifconfig                             // from wihtin the container - note the container has it's own eth0 stack with addr 10.88.0.6
eth0      Link encap:Ethernet  HWaddr 2A:1D:F6:6E:12:F6  
          inet addr:10.88.0.6  Bcast:10.88.255.255  Mask:255.255.0.0
          inet6 addr: fe80::281d:f6ff:fe6e:12f6/64 Scope:Link
          UP BROADCAST RUNNING MULTICAST  MTU:1500  Metric:1
          RX packets:45 errors:0 dropped:0 overruns:0 frame:0
          TX packets:9 errors:0 dropped:0 overruns:0 carrier:0
          collisions:0 txqueuelen:0 
          RX bytes:6490 (6.3 KiB)  TX bytes:682 (682.0 B)

$ sudo podman run -it alpine sh         // create a second root full container, since sudo is used.
/# ifconfig eth0                        // from wihtin the container - note the container has it's own eth0 stack with addr 10.88.0.8
eth0      Link encap:Ethernet  HWaddr 32:34:03:CD:C9:BE  
          inet addr:10.88.0.8  Bcast:10.88.255.255  Mask:255.255.0.0
          inet6 addr: fe80::3034:3ff:fecd:c9be/64 Scope:Link
          UP BROADCAST RUNNING MULTICAST  MTU:1500  Metric:1
          RX packets:20 errors:0 dropped:0 overruns:0 frame:0
          TX packets:6 errors:0 dropped:0 overruns:0 carrier:0
          collisions:0 txqueuelen:0 
          RX bytes:2828 (2.7 KiB)  TX bytes:472 (472.0 B)

/# ifconfig 10.88.0.8

$ sudo podman network create --help | more
$ sudo podman network create podman1
$ /etc/cni/net.d/podman1.conflist
$ sudo podman network ls
$ NETWORK ID    NAME         VERSION     PLUGINS
$ 2f259bab93aa  podman       0.4.0       bridge,portmap,firewall,tuning
$ b932778640d3  cni-podman1  0.4.0       bridge,portmap,firewall,tuning,dnsname
$ 25f76a05ffcd  podman1      0.4.0       bridge,portmap,firewall,tuning,dnsname
$ sudo podman run --net=podman1 -it alpine sh
$ sudo podman netowrk connect --help | more
$ sudo podman ps
$ CONTAINER ID  IMAGE                            COMMAND     CREATED        STATUS            PORTS       NAMES
$ f8738f7dacb4  docker.io/library/alpine:latest  sh          4 minutes ago  Up 4 minutes ago              goofy_boyd
$ sudo podman inspect goofy_boyd | grep IPA
$            "IPAddress": "",
$                    "IPAddress": "10.89.1.2",
$                    "IPAMConfig": null,
$ sudo podman network connect podman goofy_boyd
$ sudo podman inspect goofy_boyd | grep IPA
$            "IPAddress": "",
$                    "IPAddress": "10.88.0.9",
$                    "IPAMConfig": null,
$                    "IPAddress": "10.89.1.2",
$                    "IPAMConfig": null,
$ sudo podman exec -it goofy_boyd sh
/ # ifconfig                                     // this container has eth0 stack with ip address 10.89.1.2 and eth1 stack with ip address 10.88.0.9
eth0      Link encap:Ethernet  HWaddr BE:8A:C4:6E:74:E4  
          inet addr:10.89.1.2  Bcast:10.89.1.255  Mask:255.255.255.0
          inet6 addr: fe80::bc8a:c4ff:fe6e:74e4/64 Scope:Link
          UP BROADCAST RUNNING MULTICAST  MTU:1500  Metric:1
          RX packets:128 errors:0 dropped:0 overruns:0 frame:0
          TX packets:13 errors:0 dropped:0 overruns:0 carrier:0
          collisions:0 txqueuelen:0 
          RX bytes:20644 (20.1 KiB)  TX bytes:962 (962.0 B)

eth1      Link encap:Ethernet  HWaddr 12:73:87:00:E2:25  
          inet addr:10.88.0.9  Bcast:10.88.255.255  Mask:255.255.0.0
          inet6 addr: fe80::1073:87ff:fe00:e225/64 Scope:Link
          UP BROADCAST RUNNING MULTICAST  MTU:1500  Metric:1
          RX packets:47 errors:0 dropped:0 overruns:0 frame:0
          TX packets:11 errors:0 dropped:0 overruns:0 carrier:0
          collisions:0 txqueuelen:0 
          RX bytes:6872 (6.7 KiB)  TX bytes:822 (822.0 B)

$ sudo podman run --net=host -it alpine sh

NOTE:
	MACVLAN network and overlay network

$ sudo podman network create --driver macvlan sublet=172.31.10.0/16 -gateway=172.31.0.1 -o parent=eth0 mynet

NOTE:
	Podman Network with rootless containers.

$ podman info | grep -A3 -i network
$  network:
$  - bridge
$  - macvlan
$  volume:

$ podman network ls
$ NETWORK ID    NAME        VERSION     PLUGINS
$ 2f259bab93aa  podman      0.4.0       bridge,portmap,firewall,tuning
$ podman network inspect podman
$        "cniVersion": "0.4.0",
$        "name": "podman",
$        "plugins": [
$            {
$                "bridge": "cni-podman0",
$                "hairpinMode": true,
$                "ipMasq": true,
$                "ipam": {
$                    "ranges": [
$                        [
$                            {
$                                "gateway": "10.88.0.1",
$                                "subnet": "10.88.0.0/16"
$                            }
$ 
$ podman run -it --network=podman docker.io/alpine sh
$ podman run -it --network=podman quay.io/libpod/banner sh
$ podman run -it -d --name=web --network=podman quay.io/libpod/banner
$ podman ps -a
$ CONTAINER ID  IMAGE                            COMMAND               CREATED         STATUS                      PORTS       NAMES
$ 976bbd5fcec1  quay.io/libpod/banner:latest     nginx -g daemon o...  51 seconds ago  Up 51 seconds ago                       web
$ dd91da90830f  docker.io/library/alpine:latest  sh                    9 seconds ago   Exited (127) 2 seconds ago              serene_gagarin
$ podman inspect serene_gagarin | grep IPA
$            "IPAddress": "",
$ podman inspect web | grep IPA
$            "IPAddress": "",
$                    "IPAddress": "10.88.0.2",
$                    "IPAMConfig": null,
/ # apk --update add curl
$ curl http://10.88.0.2
#### HOME
$ podman images
$ REPOSITORY  TAG         IMAGE ID      CREATED       SIZE
$ <none>      <none>      d6e77351e477  2 months ago  1.8 GB
$ podman volume --help
$ podman volume create mydata
$ podman volume inspect mydata
$ podman run -it --name=test-con1 -v mydata:/opt/data d6e77351e477 bash
# df -h
# Filesystem                              Size  Used Avail Use% Mounted on
# fuse-overlayfs                           76G   36G   40G  48% /
# tmpfs                                    64M     0   64M   0% /dev
# /dev/mapper/rhel_vdi--rhel8--base-root   76G   36G   40G  48% /opt/data
# shm                                      63M     0   63M   0% /dev/shm
# tmpfs                                   3.2G  376K  3.2G   1% /etc/hosts
# tmpfs                                    16G     0   16G   0% /sys/fs/cgroup
# devtmpfs                                 16G     0   16G   0% /dev/tty
# tmpfs                                    16G     0   16G   0% /proc/acpi
# tmpfs                                    16G     0   16G   0% /proc/scsi
# tmpfs                                    16G     0   16G   0% /sys/firmware
# tmpfs                                    16G     0   16G   0% /sys/fs/selinux
# tmpfs                                    16G     0   16G   0% /sys/dev/block

$ podman pod --help
$ podman pod create --name=my_pod
$ podman pod inspect my_pod
$ podman images
$ REPOSITORY              TAG               IMAGE ID      CREATED        SIZE
$ localhost/podman-pause  4.4.1-1686839996  d00829338d48  2 minutes ago  735 kB
$ podman pod ps
$ POD ID        NAME        STATUS      CREATED        INFRA ID      # OF CONTAINERS
$ ae6275207f0e  my_pod      Created     2 minutes ago  e61fde949fe5  1
$ podman ps -a
$ CONTAINER ID  IMAGE                                    COMMAND     CREATED         STATUS                      PORTS       NAMES
$ c5280fceb6f3  d6e77351e477                             bash        14 minutes ago  Exited (127) 5 minutes ago              test-con1
$ e61fde949fe5  localhost/podman-pause:4.4.1-1686839996              3 minutes ago   Created                                 ae6275207f0e-infr
$ podman ps -a
$ CONTAINER ID  IMAGE                                    COMMAND     CREATED         STATUS                     PORTS       NAMES
$ e61fde949fe5  localhost/podman-pause:4.4.1-1686839996              7 minutes ago   Up 56 seconds                          ae6275207f0e-infra
$ 212d955b09f6  d6e77351e477                             bash        55 seconds ago  Exited (0) 55 seconds ago              reverent_mahavira
$ podman run -d --pod=my_pod d6e77351e477
$ podman ps -a --pod
$ CONTAINER ID  IMAGE                                    COMMAND     CREATED        STATUS                    PORTS       NAMES               POD ID        PODNAME
$ e61fde949fe5  localhost/podman-pause:4.4.1-1686839996              9 minutes ago  Up 2 minutes                          ae6275207f0e-infra  ae6275207f0e  my_pod
$ 212d955b09f6  d6e77351e477                             bash        2 minutes ago  Exited (0) 2 minutes ago              reverent_mahavira   ae6275207f0e  my_pod
$ podman run -it -d --pod=my_pod d6e77351e477 bash
$ podman ps -a --pod
$ CONTAINER ID  IMAGE                                    COMMAND     CREATED         STATUS                    PORTS       NAMES               POD ID        PODNAME
$ e61fde949fe5  localhost/podman-pause:4.4.1-1686839996              11 minutes ago  Up 5 minutes                          ae6275207f0e-infra  ae6275207f0e  my_pod
$ 212d955b09f6  d6e77351e477                             bash        5 minutes ago   Exited (0) 5 minutes ago              reverent_mahavira   ae6275207f0e  my_pod
$ 14d661c923f6  d6e77351e477                             bash        6 seconds ago   Up 6 seconds                          upbeat_beaver       ae6275207f0e  my_pod
$ podman pod ps
$ POD ID        NAME        STATUS      CREATED         INFRA ID      # OF CONTAINERS
$ ae6275207f0e  my_pod      Degraded    12 minutes ago  e61fde949fe5  3
$ podman exec -it upbeat_beaver bash
# dnf install -y net-tools
# ifconfig eth0
$ podman inspect upbeat_beaver | grep IPA
$               "IPAddress": "",
$ podman ps --ns
$ CONTAINER ID  NAMES               PID         CGROUPNS    IPC         MNT         NET         PIDNS       USERNS      UTS
$ e61fde949fe5  ae6275207f0e-infra  92542       4026531835  4026532788  4026532785  4026532629  4026532789  4026532626  4026532787
$ 14d661c923f6  upbeat_beaver       92870       4026531835  4026532788  4026532790  4026532629  4026532791  4026532626  4026532787
$ podman pod ps
$ podman pod rm my_pod
$ podman system prune
$ podman system prune --all -f --volumes
$ podman system df
$ podman pod create --name=wp -p 38080:80     //port forwarding
$ podman pod ps
$ POD ID        NAME        STATUS      CREATED        INFRA ID      # OF CONTAINERS
$ 6edacbdd2729  wp          Created     4 seconds ago  a63404f4ec6c  1
$ podman images
$ REPOSITORY              TAG               IMAGE ID      CREATED             SIZE
$ localhost/podman-pause  4.4.1-1686839996  b0d02dc88804  About a minute ago  735 kB
$ podman load --input=./<some-image>
$ podman images
$ REPOSITORY              TAG               IMAGE ID      CREATED        SIZE
$ localhost/podman-pause  4.4.1-1686839996  b0d02dc88804  2 minutes ago  735 kB
$ <none>                  <none>            74223a7464ea  6 weeks ago    1.45 GB
$ podman run -d --restart=always --pod=wp --name=wp_testdb 74223a7464ea
$ podman ps -a
$ CONTAINER ID  IMAGE                                    COMMAND     CREATED         STATUS                 PORTS                  NAMES
$ a63404f4ec6c  localhost/podman-pause:4.4.1-1686839996              5 minutes ago   Up 19 seconds          0.0.0.0:38080->80/tcp  6edacbdd2729-infra
$ de459e974f59  74223a7464ea                             bash        19 seconds ago  Up Less than a second  0.0.0.0:38080->80/tcp  wp_testd
$ podman ps -a --pod
$ CONTAINER ID  IMAGE                                    COMMAND     CREATED             STATUS             PORTS                  NAMES               POD ID        PODNAME
$ a63404f4ec6c  localhost/podman-pause:4.4.1-1686839996              6 minutes ago       Up About a minute  0.0.0.0:38080->80/tcp  6edacbdd2729-infra  6edacbdd2729  wp
$ de459e974f59  74223a7464ea                             bash        About a minute ago  Initialized        0.0.0.0:38080->80/tcp  wp_testdb           6edacbdd2729  wp
$ podman run -d --restart=always --pod=wp --name=wp_testapp 74223a7464ea
$ podman pod ps
$ POD ID        NAME        STATUS      CREATED        INFRA ID      # OF CONTAINERS
$ 6edacbdd2729  wp          Degraded    9 minutes ago  a63404f4ec6c  3
$ podman ps -a
$ CONTAINER ID  IMAGE                                    COMMAND     CREATED         STATUS                             PORTS                  NAMES
$ a63404f4ec6c  localhost/podman-pause:4.4.1-1686839996              9 minutes ago   Up 4 minutes                       0.0.0.0:38080->80/tcp  6edacbdd2729-infra
$ de459e974f59  74223a7464ea                             bash        4 minutes ago   Exited (0) Less than a second ago  0.0.0.0:38080->80/tcp  wp_testdb
$ 4f0c8b99880c  74223a7464ea                             bash        37 seconds ago  Exited (0) Less than a second ago  0.0.0.0:38080->80/tcp  wp_testapp
$ podman generate --help
$ podman generate kube wp > wp.yaml
$ podman system prune --all -f --volumes
$ podman play kube wp.yaml









