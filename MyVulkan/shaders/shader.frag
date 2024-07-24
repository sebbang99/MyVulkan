#version 450

layout(binding = 1) uniform sampler2D texSampler;

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord_kk;

layout(location = 0) out vec4 outColor;	// location ���� frame buffer index.

void main() {	// �� fragment�� ������ �����ϴµ�, vertex shader���� �� vertex�� ���� �����س��Ҵ�.
	//outColor = vec4(fragTexCoord, 0.0, 1.0); // for debugging
	outColor = texture(texSampler, fragTexCoord_kk);
}