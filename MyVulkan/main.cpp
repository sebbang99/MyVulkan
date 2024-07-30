/*
 *
 *  Copyright (c) 2024-2025, Sehee Cho
 *
 */

#include "raytracingBasic.h"

int main() {
	RaytracingBasic app;

	try {
		app.run();
	}
	catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}