FROM tensorflow:latest

WORKDIR /app

RUN apk add --no-cache \
    build-base \
    cmake \
    ninja \
    pkgconfig \
    linux-headers \
    libffi-dev \
    openssl-dev \
    hdf5-dev \
    openblas-dev \
    gcompat

RUN echo "#include <sys/types.h>" >> /usr/include/unistd.h && \
    echo "typedef int64_t off64_t;" >> /usr/include/unistd.h

RUN apk add --no-cache build-base


COPY req.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r req.txt

COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]