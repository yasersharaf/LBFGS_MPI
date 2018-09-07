################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/solvers/LBFGSSolver.cpp \
../src/solvers/QuasiNewtonSolver.cpp 

OBJS += \
./src/solvers/LBFGSSolver.o \
./src/solvers/QuasiNewtonSolver.o 

CPP_DEPS += \
./src/solvers/LBFGSSolver.d \
./src/solvers/QuasiNewtonSolver.d 


# Each subdirectory must supply rules for building sources it contributes
src/solvers/%.o: ../src/solvers/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: Cross G++ Compiler'
	mpic++ -std=c++0x -O3 -Wall -c -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


