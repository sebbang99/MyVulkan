//#include <vulkan/vulkan.h>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <iostream>		// error reporting
#include <stdexcept>	// error reporting
#include <cstdlib>		// EXIT macros
#include <vector>		// Extension Details

const uint32_t WIDTH = 987;
const uint32_t HEIGHT = 654;

class HelloTriangleApplication {
public:
	void run() {
		initWindow();
		initVulkan();
		mainLoop();
		cleanup();
	}

private:
	GLFWwindow* window;
	VkInstance instance;

	void initWindow() {
		glfwInit();

		// (�ɼǸ�, �ɼǰ�)
		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);	// OpenGL�� �⺻�̶� OpenGL �ƴ϶�� ���.
		glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

		window = glfwCreateWindow(WIDTH, HEIGHT, "Sehee's Vulkan(987 x 654)", nullptr, nullptr);	// 5th : OpenGL������ �԰���.
	}

	void initVulkan() {
		createInstance();
	}

	void mainLoop() {	// iterate until window is closed.
		while (!glfwWindowShouldClose(window)) {
			glfwPollEvents();
		}
	}

	void cleanup() {
		vkDestroyInstance(instance, nullptr);

		glfwDestroyWindow(window);

		glfwTerminate();
	}

	void createInstance() {
		// Optional, but driver�� �� specific app�� ���� optimize �ϰ� ����.
		VkApplicationInfo appInfo{};
		appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		appInfo.pApplicationName = "Hello, Sehee's Triangle";
		appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0); // (major, minor, patch)
		appInfo.pEngineName = "No Engine";
		appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.apiVersion = VK_API_VERSION_1_0;

		// Not optional, driver���� global extension�� validation layer�� ���� ����.
		VkInstanceCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		createInfo.pApplicationInfo = &appInfo;

		uint32_t glfwExtensionCount = 0;
		const char** glfwExtensions;

		glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
		
		createInfo.enabledExtensionCount = glfwExtensionCount;
		createInfo.ppEnabledExtensionNames = glfwExtensions;

		createInfo.enabledLayerCount = 0;

		// Ready for create new instance !!
		if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
			throw std::runtime_error("failed to create instance :(");
		}

		// Checking for extension details
		uint32_t extensionCount = 0;
		vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);

		std::vector<VkExtensionProperties> extensions(extensionCount);
		
		vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());

		if (checkAllExtensionsIncluded(glfwExtensionCount, glfwExtensions, extensionCount, extensions)) {
			std::cout << "All required extensions are included :-)\n";
		}
		else {
			std::cout << "Some extensions are not included :-(\n";
		}
	}

	// self-made
	bool checkAllExtensionsIncluded(uint32_t glfwExtensionCount, const char** glfwExtensions, 
		uint32_t extensionCount, std::vector<VkExtensionProperties> extensions) {
		
		// Print required extensions.
		std::cout << "required extensions:\n";

		for (int i = 0; i < glfwExtensionCount; i++) {
			std::cout << '\t' << glfwExtensions[i] << "\n";
		}

		// Print available extensions.
		std::cout << "available extensions:\n";

		for (const auto& extension : extensions) {
			std::cout << '\t' << extension.extensionName << "\n";
		}

		// Check whether all extensions are included or not.
		for (int i = 0; i < glfwExtensionCount; i++) {

			bool flag = false;
			for (const auto& extension : extensions) {
				if (!std::strcmp(glfwExtensions[i], extension.extensionName)) {
					flag = true;
					break;
				}
			}

			if (!flag) {
				return false;
			}
		}
		return true;
	}
};

int main() {
	HelloTriangleApplication app;

	try {
		app.run();
	}
	catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}