Shader "Custom/Blinn-phong"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
        _DiffusePower("Diffuse Power", Float) = 1.0
        _SpecularTex("Specular Tex", 2D) = "white"{}
        _Gloss("Specular Gloss", Float) = 0.5
        _SpecularColor("Specular Color", Color) = (1,1,1,1)

    }
    SubShader
    {
        Cull off
        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "UnityCG.cginc"
            #include "Lighting.cginc"
            #include "AutoLight.cginc"
            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
                float3 normal : NORMAL;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
                float3 normalDir : TEXCOORD1;
                float4 posWorld : TEXCOORD2;
            };

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv;
                o.vertex = float4(v.uv, 0, 1);
                o.vertex.y = 1 - o.vertex.y;
                
                o.vertex.xy= o.vertex.xy*2-1;
                //o.vertex = UnityObjectToClipPos(o.vertex);
                // 将物体法线从物体坐标系转换到世界坐标系
                o.normalDir = UnityObjectToWorldNormal(v.normal);

                o.posWorld = mul(unity_ObjectToWorld, v.vertex);
                return o;
            }

            sampler2D _MainTex;
            float _DiffusePower;
            sampler2D _SpecularTex;
            float _Gloss;
            float4 _SpecularColor;

            fixed4 frag (v2f i) : SV_Target
            {
                // 法线方向
                float3 normalDirection = normalize(i.normalDir);
                // 灯光方向
                float lightDirection = normalize(_WorldSpaceLightPos0.xyz);
                // 灯光颜色
                float3 lightColor = _LightColor0.rgb;
                // 视线方向
                float3 viewDirection = normalize(_WorldSpaceCameraPos.xyz - i.posWorld.xyz);
                // 视线方先与法线方向的中间向量
                float3 halfDirection = normalize(viewDirection+lightDirection);

                // 计算灯光衰减
                float attenuation = LIGHT_ATTENUATION(i);
                float3 attenColor = attenuation * _LightColor0.xyz;

                // 基于兰伯特模型计算漫反射灯光
                float NdotL = max(0,dot(normalDirection,lightDirection));
                // 方向光
                float3 directionDiffuse = pow(NdotL, _DiffusePower) * attenColor;
                // 环境光  
                float3 inDirectionDiffuse = float3(0,0,0)+UNITY_LIGHTMODEL_AMBIENT.rgb;


                // 基于Blinn Phong计算镜面反射灯光
                float specPow = exp2( _Gloss * 10.0 + 1.0 );
                float3 directionSpecular = attenColor*  _SpecularColor*(pow(max(0,dot(halfDirection,normalDirection)),specPow));

                // 灯光与材质球表面颜色进行作用
                float3 texColor = tex2D(_MainTex, i.uv).rgb;
                float3 diffuseColor = texColor *(directionDiffuse+inDirectionDiffuse);
                float3 specularColor = directionSpecular;
                float4 finalColor = float4(diffuseColor+specularColor,1);

                return finalColor;
            }
            ENDCG
        }
    }
}