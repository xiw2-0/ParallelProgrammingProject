SRC_DIR = ./src
BUILD_DIR = ./build
TEST_DIR = ./test

ccsrc = ${wildcard ${SRC_DIR}/*.cc}
cctest = ${wildcard ${TEST_DIR}/*.cc}

ccobj = ${patsubst ${SRC_DIR}%, ${BUILD_DIR}%, ${ccsrc:.cc=.o}}
cctestobj = ${patsubst ${TEST_DIR}%, ${BUILD_DIR}%, ${cctest:.cc=.o}}

mainobj = %main_test.o
obj = ${filter-out ${mainobj}, ${ccobj} ${cctestobj}}

TEST = ${BUILD_DIR}/test
test_obj = ${BUILD_DIR}/main_test.o

INC_DIR = -I${SRC_DIR}
CCFLAGS = ${INC_DIR} -std=c++11 -g
LDFLAGS = -lpthread


all: ${TEST}
.PHONY: all

${TEST}: ${test_obj} $(obj)
	$(CXX) $^ -o $@  $(LDFLAGS)

${BUILD_DIR}/%.o: ${TEST_DIR}/%.cc
	$(CXX) -c $< -o $@ ${CCFLAGS}

${BUILD_DIR}/%.o: ${SRC_DIR}/%.cc
	$(CXX) -c $< -o $@ ${CCFLAGS}


.PHONY: clean
clean:
	rm -f $(obj) ${test_obj} \
				${TEST}
