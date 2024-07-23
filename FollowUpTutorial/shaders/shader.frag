#version 450

layout(binding = 1) uniform sampler2D texSampler;

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord_kk;

layout(location = 0) out vec4 outColor;	// location 값은 frame buffer index.

void main() {	// 각 fragment의 색깔을 지정하는데, vertex shader에서 준 vertex의 색을 보간해놓았다.
	//outColor = vec4(fragTexCoord, 0.0, 1.0); // for debugging
	outColor = texture(texSampler, fragTexCoord_kk);
}