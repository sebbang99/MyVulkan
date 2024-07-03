#version 450

// vertex shader에서 ubo descriptor를 통해 VkBuffer에 접근할 수 있다.
layout(binding = 0) uniform UniformBufferObject {	
	mat4 model;
	mat4 view;
	mat4 proj;
} ubo;

layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec3 inColor;

layout(location = 0) out vec3 fragColor;

void main() {
	gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 0.0, 1.0);
	fragColor = inColor;
}