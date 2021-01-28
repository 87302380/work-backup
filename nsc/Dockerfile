FROM ubuntu:latest
ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Set env
ENV NS3_VERSION=ns-3.30
ENV NSC_VERSION=nsc-0.5.3
ENV BUILD=/build

# build ns3
WORKDIR $BUILD/ns3
RUN apt-get update
# Minimal dependencies
RUN  apt-get install -y g++\
    make\
    cmake\
    python3\
    python3-dev\
    pkg-config\
    sqlite3\
    python3-distro\
    python3-requests\
    gdb

# python bindings dependencies
RUN  apt-get install -y python3-setuptools git
RUN git clone https://gitlab.com/nsnam/bake /build/ns3


# build nsc
RUN apt-get install -y flex-old bison wget python
WORKDIR /build/nsc
RUN wget http://research.wand.net.nz/software/nsc/$NSC_VERSION.tar.bz2
RUN tar xf $NSC_VERSION.tar.bz2
RUN cd $NSC_VERSION && patch -p1 < $BUILD/pak/patches/nsc.patch
RUN cd $NSC_VERSION && ./scons.py


