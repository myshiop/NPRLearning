// Upgrade NOTE: replaced 'mul(UNITY_MATRIX_MVP,*)' with 'UnityObjectToClipPos(*)'

Shader "NPR/Outline"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
        _OutlineColor ("OutlineColor", Color) = (1,1,1,1)
        _OutlineWidth ("OutlineWidth", Float) = 0.1
        _MainColor ("Main Color", Color) = (1.0, 1.0, 1.0, 1.0)//主颜色，默认白色
        _SpecularColor ("Specular Color", Color) = (0, 0, 0, 1.0)//高光颜色，默认黑色
        _Shininess ("Gloss", Range(0.0, 10)) = 0.5//反光度    
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 100

        Pass
        {
            Tags { "LightMode" = "ForwardBase" }

            CGPROGRAM

            #pragma vertex vert
            #pragma fragment frag
            
            #include "UnityCG.cginc"
            #include "Lighting.cginc" 

            struct a2v
            {
                float4 vertex : POSITION;//顶点
                float2 uv : TEXCOORD0;//uv
                float3 normal : NORMAL;//法线
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;//顶点
                float3 worldLightDir:TEXCOORD1;//世界坐标系下的指向光源的矢量
                float3 worldNormal:TEXCOORD2;//世界坐标系下法线
                float3 worldViewDir :TEXCOORD3; //世界坐标系下的指向观察者的矢量
                float4 pos : SV_POSITION;//裁剪坐标下的顶点
            };

            sampler2D _MainTex;
            float4 _MainTex_ST;
            fixed4 _SpecularColor;
            fixed4 _MainColor;
            float _Shininess;
            
            v2f vert (a2v v)
            {
                v2f o;

                //使用UNITY_MATRIX_MVP矩阵做仿射变换，把模型空间下的顶点转到裁剪坐标下
                o.pos = UnityObjectToClipPos(v.vertex);
                
                //取得世界坐标系下的法线,UnityObjectToWorldNormal()在UnityCG.cginc被定义
                o.worldNormal = UnityObjectToWorldNormal(v.normal);

                //取得世界坐标系下的指向光源的矢量，WorldSpaceLightDir()在UnityCG.cginc被定义
                o.worldLightDir = WorldSpaceLightDir(v.vertex);

                //取得世界坐标系下的指向观察者的矢量，WorldSpaceLightDir()在UnityCG.cginc被定义
                o.worldViewDir = WorldSpaceViewDir(v.vertex);

                //uv采样
                o.uv = TRANSFORM_TEX(v.uv, _MainTex);
                return o;
            }
            
            fixed4 frag (v2f i) : SV_Target
            {
                //归一化
                fixed3 normalizedLightDir  = normalize(i.worldLightDir);
                fixed3 normalizedNormal = normalize(i.worldNormal);
                fixed3 normalizedViewDir = normalize(i.worldViewDir);

                //像素颜色采样
                fixed3 albedo = tex2D(_MainTex, i.uv);
                
                //计算环境光
                fixed3 ambient = UNITY_LIGHTMODEL_AMBIENT.xyz * albedo;

                //计算漫反射
                fixed3 diffuse = (_LightColor0.rgb * albedo) * saturate(dot(normalizedNormal,normalizedLightDir));

                //计算高光
                fixed3 halfDir = normalize(normalizedViewDir + normalizedLightDir);
                fixed3 specular = (_SpecularColor.rgb * _LightColor0.rgb) * pow(saturate(dot(halfDir,normalizedNormal )),_Shininess);

                return fixed4((ambient+diffuse+specular),1);
            }
            ENDCG
        }

        Pass
        {
            Name "OutLine"
            Tags{"lightmode" = "Always"}
            Cull Front
            ZWrite On
            ColorMask RGB
            Blend SrcAlpha OneMinusSrcAlpha

            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"

            struct appdata{
                float4 vertex : POSITION;
                float4 normal : NORMAL;
                float4 texCoord : TEXCOORD0;
            };

            struct v2f{
                float4 pos : SV_POSITION;
                float4 color : COLOR;
                float4 tex : TEXCOORD0;
            };

            float2 hash22(float2 p) {
                p = float2(dot(p, float2(127.1, 311.7)), dot(p, float2(269.5, 183.3)));
                return -1.0 + 2.0 * frac(sin(p) * 43758.5453123);
            }

            float2 hash21(float2 p) {
                float h = dot(p, float2(127.1, 311.7));
                return -1.0 + 2.0 * frac(sin(h) * 43758.5453123);
            }

            //perlin
            float perlin_noise(float2 p) {
                float2 pi = floor(p);
                float2 pf = p - pi;
                float2 w = pf * pf * (3.0 - 2.0 * pf);
                return lerp(lerp(dot(hash22(pi + float2(0.0, 0.0)), pf - float2(0.0, 0.0)),
                dot(hash22(pi + float2(1.0, 0.0)), pf - float2(1.0, 0.0)), w.x),
                lerp(dot(hash22(pi + float2(0.0, 1.0)), pf - float2(0.0, 1.0)),
                dot(hash22(pi + float2(1.0, 1.0)), pf - float2(1.0, 1.0)), w.x), w.y);
            }

            float4 _OutlineColor;
            float _OutlineWidth;
            uniform half4 _NoiseTillOffset;
            uniform half _NoiseAmp;

            v2f vert(appdata v){
                v2f o;
                o.pos = UnityObjectToClipPos(v.vertex);
                float3 norm = mul((float3x3)UNITY_MATRIX_IT_MV, v.normal);
                float2 extendDir = normalize(TransformViewToProjection(norm.xy));
                
                float2 noiseSampleTex = v.texCoord;
                noiseSampleTex = noiseSampleTex * _NoiseTillOffset.xy + _NoiseTillOffset.zw;
                float noiseWidth = perlin_noise(noiseSampleTex);
                noiseWidth = noiseWidth * 2 - 1;

                half outlineWidth = _OutlineWidth + _OutlineWidth * noiseWidth;

                o.pos.xy += extendDir * (o.pos.w * _OutlineWidth * 0.1);
    
                o.tex = v.texCoord;
                o.color = _OutlineColor;
                return o;
            }

            half4 frag(v2f i):SV_TARGET{
                return i.color;
            }
            ENDCG
        }
    }
}
