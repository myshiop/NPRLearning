Shader "NPR/Lambert"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
        _LightMap ("lightmap", 2D) = "white" {}

        _Color ("Color", Color) = (1, 1, 1, 1)
        _ShadowAttWeight ("ShadowAttWeight", Range(0, 0.5)) = 0.3
        _DividLineM ("DividLineM", Range(-0.5, 0.8)) = 0.0
        _DividLineD ("DividLineD", Range(-1, -0.0)) = -0.5
        _DividLineSpec ("_DividLineSpec", Range(0.5, 1)) = 0.8
        _Atten ("atten", Range(0, 1)) = 0.1
        _FresnelEff ("FresnelEff", Range(0, 1)) = 0.5
        _Glossiness ("Glossiness", Range(0, 1)) = 0.1
        _DividSharpness("DividSharpness", Range(0.2, 5)) = 1.0 //控制sigmoid过渡带宽的变量
        _DarkFaceColor("DarkFaceColor", Color) = (1, 1, 1, 1) //第二层暗面颜色调整
        _DeepDarkColor("DeepDarkColor", Color) = (1, 1, 1, 1) //第三层暗面颜色调整
        _diffuseBright("diffuseBright", Range(0.0, 2.0)) = 1.0 //wrap函数控制变量，用来抬高输出值
        _FresnelColor("FresnelColor", Color) = (1, 1, 1, 1)//控制fresnel的颜色变量
        _AOWeight("AOWeight", Range(0.0, 2.0)) = 1.0 //控制ao的变量

        [Toggle(ENABLE_SSS)] _SSS("SSS", Range(0,1)) = 0.0
        _Radius("Radius", Range(0, 1)) = 0.1

        //SSS
        _SSSColor ("SSSColor", Color) = (1, 0, 0, 1)
        _SSSWeight ("SSSWeight", Range(0, 1)) = 0.0
        _SSSSize ("SSSSize", Range(0, 1)) = 0.0
        _SSForwardAtt("Att of SS in forward Dir", Range(0,1)) = 0.5


        //描边用
        _OutlineColor ("OutlineColor", Color) = (1,1,1,1)
        _OutlineWidth ("OutlineWidth", Range(0, 0.5)) = 0.024
    }
    SubShader 
    {
        Tags { "RenderType"="Opaque" }

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma shader_feature ENABLE_SSS

            #include "UnityCG.cginc"
            #include "Lighting.cginc"
            #include "AutoLight.cginc"
            
            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
                float2 uv_lightMap : TEXCOORD1;
                float3 normal : NORMAL;
            };
            
            struct v2f
            {
                float2 uv : TEXCOORD0;
                float2 uv_lightMap : TEXCOORD1;
                float4 pos : SV_POSITION;
                float3 worldNormal : TEXCOORD2;
                float3 worldPos : TEXCOORD3;
                //SHADOW_COORDS(3)
            };
            
            sampler2D _MainTex;
            sampler2D _LightMap;
            float _ShadowAttWeight;
            float _DividLineM;
            float _DividLineD;
            float _DividLineSpec;
            float _Roughness;
            fixed4 _Color;
            float4 _MainTex_ST;
            float4 _LightMap_ST;
            float atten;
            float _FresnelEff;
            float _Glossiness;
            fixed4 _DarkFaceColor;
            fixed4 _DeepDarkColor;
            fixed4 _FresnelColor;
            float4 _LightMap_TexelSize;
            half _AOWeight;
            half _DividSharpness;
            half _diffuseBright;

            half _Radius;
            float _SSS;

            //SSS
            fixed4 _SSSColor;
		    half _SSSWeight;
		    half _SSSSize;
		    half _SSForwardAtt;

            //center决定窗函数的宽度，sharp决定窗函数的平滑程度
            float sigmoid(float x, float center, float sharp){
                float s;
                s = 1 / (1 + pow(100000, (-3 * sharp * (x - center))));
                return s;
            }

            half Pow2(half x){
                return pow(x, 2);
            }

            // half Pow5(half x){
            //     return pow(x, 5);
            // }

            half Pow3(half x){
                return pow(x, 3);
            }

            float ndc2Normal(float x){
                return x * 0.5 + 0.5;
            }

            float Normal2ndc(float x){
                return x * 2 - 1;
            }

            float D_GGX(float a2, float NoH){
                float d = (NoH * a2 - NoH) * NoH + 1;
                return a2 / (3.1415926 * d * d); 
            }

            float3 warp(float3 x, float3 w){
                return (x + w) / (1 + w);
            }

            float3 Fresenl_schlick(float VoN, float3 rF0){
                return rF0 + (1 - rF0) * Pow5(1 - VoN);
            }

            float3 Fresenl_extend(float VoN, float3 rF0){
                return rF0 + (1 - rF0) * Pow3(1 - VoN);
            }

            float Gaussion(float x, float center, float var){
                return pow(2.718, -1 * Pow2(x - center) / var);
            }

            half GaussionSimple(half x, half var){
                return 1.0 / (var * sqrt(2 * 3.14)) * pow(2.718, (-1 * x * x) / (2 * var * var)); 
            }

            half3 diffuse_profile(half d){
                half3 n1 = (0.233,0.455,0.649);
                half3 n2 = (0.100,0.336,0.344);
                half3 n3 = (0.118,0.198,0.000);
                half3 n4 = (0.113,0.007,0.007);
                half3 n5 = (0.358,0.004,0.000);
                half3 n6 = (0.078,0.000,0.000);

                half3 result = n1 * GaussionSimple(d, 0.0064) + n2 * GaussionSimple(d, 0.0484) + n3 * GaussionSimple(d, 0.1870) + n4 * GaussionSimple(d, 0.5670) + n5 * GaussionSimple(d, 1.9900) + n6 * GaussionSimple(d, 7.4100); 

                return result;
            }

            half3 IntegratePre(v2f i, half radius){
                half x = -3.14;
                half delta = acos(dot(normalize(i.worldNormal), normalize(UnityWorldSpaceViewDir(i.worldPos))));
                half3 totalWeight = (0.0, 0.0, 0.0);
                half3 totalLight = (0.0, 0.0, 0.0);
                while (x < 3.14)
                {
                    half simpleAngle = delta + x;
                    half sampleDist = abs(2.0 * radius * sin(x * 0.5));
                    half diff = max(cos(simpleAngle), 0.0);
                    half weight = diffuse_profile(sampleDist);
                    totalLight += weight * diff;
                    totalWeight += weight;
                    x += 0.1;
                }

                half3 result = totalLight / totalWeight;

                return result;    
            }

            fixed4 Tex2DLightMap(v2f s){
                fixed4 ilm = tex2D(_LightMap, s.uv_lightMap);

                //blur
                float2 tmpuv1 = s.uv_lightMap + _LightMap_TexelSize.xy;
                float2 tmpuv2 = s.uv_lightMap - _LightMap_TexelSize.xy;
                float2 tmpuv3 = s.uv_lightMap;
                tmpuv3.x += _LightMap_TexelSize.x;
                tmpuv3.y -= _LightMap_TexelSize.y;
                float2 tmpuv4 = s.uv_lightMap;
                tmpuv4.x -= _LightMap_TexelSize.x;
                tmpuv4.y += _LightMap_TexelSize.y;

                fixed4 ilm1 = tex2D(_LightMap, tmpuv1);
                fixed4 ilm2 = tex2D(_LightMap, tmpuv2);
                fixed4 ilm3 = tex2D(_LightMap, tmpuv3);
                fixed4 ilm4 = tex2D(_LightMap, tmpuv4);

                ilm = 0.2 * (ilm + ilm1 + ilm2 + ilm3 + ilm4);

                return ilm;
            }

            half3 FresnelIntensity(v2f s, half3 lightDir, half3 viewDir){
                half3 nNormal = normalize(s.worldNormal);
                half VoN = dot(nNormal, viewDir);
                half VoL = dot(viewDir, lightDir);
                half3 fresnel = Fresenl_extend(VoN, float3(0.1, 0.1, 0.1));
                half3 fresnel_result = _FresnelEff * fresnel * (1 - VoL) / 2;

                return fresnel_result;
            }

            half DiffuseIntensity(v2f s, half3 lightDir, half atten){
                float _BoundSharp = 9.5 * Pow2(_Roughness - 1) + 0.5;
                half AO = Tex2DLightMap(s).g;

                half3 nNormal = normalize(s.worldNormal);
                half NoL = dot(nNormal, lightDir) + _ShadowAttWeight * (atten - 1);
 
                half lambert = NoL + _AOWeight * Normal2ndc(AO);

                half MidSig = sigmoid(NoL, _DividLineM, _BoundSharp * _DividSharpness);
                half DarkSig = sigmoid(NoL, _DividLineD, _BoundSharp * _DividSharpness);

                half MidLWin = MidSig;
                half MidDWin = DarkSig - MidSig;
                half DeepWin = 1 - DarkSig;

                half diffuseLumin1 = (1 + ndc2Normal(_DividLineM)) / 2;
                half diffuseLumin2 = MidDWin * (ndc2Normal(_DividLineM) + ndc2Normal(_DividLineD)) / 2;
                half diffuseLumin3 = DeepWin * (ndc2Normal(_DividLineD));

                //half Intensity = HLightWin * 1.0 + MidLWin * 0.8 + MidDWin * 0.5 + DarkWin * 0.3;
                half3 DiffIntensity = MidLWin * (diffuseLumin1).xxx
                                 + (diffuseLumin2).xxx * _DarkFaceColor.rgb * 3 / (_DarkFaceColor.r + _DarkFaceColor.g + _DarkFaceColor.b)
                                 + (diffuseLumin3).xxx * _DeepDarkColor.rgb * 3 / (_DeepDarkColor.r + _DeepDarkColor.g + _DeepDarkColor.b);
                DiffIntensity = warp(DiffIntensity, _diffuseBright.xxx);

                return DiffIntensity.x;

            }

            half3 SSS(v2f s, half3 lightDir, half atten){
                float _BoundSharp = 9.5 * Pow2(_Roughness - 1) + 0.5;
                half AO = Tex2DLightMap(s).g;

                half3 nNormal = normalize(s.worldNormal);
                half NoL = dot(nNormal, lightDir) + _ShadowAttWeight * (atten - 1);
 
                half lambert = NoL + _AOWeight * Normal2ndc(AO);

                half MidSig = sigmoid(NoL, _DividLineM, _BoundSharp * _DividSharpness);
                half DarkSig = sigmoid(NoL, _DividLineD, _BoundSharp * _DividSharpness);

                half MidLWin = MidSig;
                half MidDWin = DarkSig - MidSig;
                half DeepWin = 1 - DarkSig;

                half diffuseLumin1 = (1 + ndc2Normal(_DividLineM)) / 2;
                half diffuseLumin2 = MidDWin * (ndc2Normal(_DividLineM) + ndc2Normal(_DividLineD)) / 2;
                half diffuseLumin3 = DeepWin * (ndc2Normal(_DividLineD));

                ///SSS
                half SSMidLWin = Gaussion(lambert, _DividLineM, _SSForwardAtt * _SSSSize);
                half SSMidDWin = Gaussion(lambert, _DividLineM, _SSSSize);
                half3 SSLumin1 = (MidLWin * diffuseLumin2) * _SSForwardAtt * SSMidLWin;
                half3 SSLumin2 = ((MidDWin + DeepWin) * diffuseLumin2) * SSMidDWin;
                half3 SS = _SSSWeight * (SSLumin1 + SSLumin2) * _SSSColor.rgb;

                return SS;
            }

            half specularIntensity(v2f s, half3 lightDir, half3 viewDir, half atten){
                half smoothness = Tex2DLightMap(s).r;
                half specIntensity = Tex2DLightMap(s).b;
                half roughness = 0.95 - 0.95 * (smoothness * _Glossiness);
                half _BoundSharp = 9.5 * Pow2(roughness - 1) + 0.5;

                half3 nNormal = normalize(s.worldNormal);
                half3 HDir = normalize(lightDir + viewDir);
                half NoH = Pow2(dot(nNormal, HDir)) + _ShadowAttWeight * (atten - 1);
                half NDF0 = D_GGX(roughness * roughness, 1);
                half NDF_HBound = NDF0 * _DividLineSpec;
                half NDF = D_GGX(roughness * roughness, clamp(0, 1, NoH));

                half specularWin = sigmoid(NDF, NDF_HBound, _BoundSharp * _DividSharpness);


                half Intensity = specularWin * (NDF0 + NDF_HBound) / 2 * specIntensity;

                return Intensity;
            }

            v2f vert(appdata v)
            {
                v2f o;
                o.pos = UnityObjectToClipPos(v.vertex);
                o.uv = TRANSFORM_TEX(v.uv, _MainTex);
                o.uv_lightMap = TRANSFORM_TEX(v.uv_lightMap, _LightMap);
                o.worldNormal = UnityObjectToWorldNormal(v.normal);
                o.worldPos = mul(unity_ObjectToWorld, v.vertex);
                
                return o;
            }
            
            fixed4 frag(v2f i) : SV_Target
            {
                half3 lightDir = normalize(UnityWorldSpaceLightDir(i.worldPos));
                half3 viewDir = normalize(UnityWorldSpaceViewDir(i.worldPos));
                float atten = 0.1;

                //计算反射率
                half diff = DiffuseIntensity(i, lightDir, atten);
                fixed3 albedo = tex2D(_MainTex, i.uv).rgb * _Color.rgb;
                fixed3 diffuse = albedo * _LightColor0.rgb;
                diffuse =  diffuse * diff.xxx * _LightColor0.rgb;

                //计算环境光
                fixed3 ambient = UNITY_LIGHTMODEL_AMBIENT.xyz * albedo;

                //计算高光
                half spec = specularIntensity(i, lightDir, viewDir, atten);
                fixed3 specular = spec * _LightColor0.rgb;

                //计算fresnel
                half3 fresnel = FresnelIntensity(i, lightDir, viewDir) * _FresnelColor.rgb;

                //计算SSS
                #if ENABLE_SSS
                    half3 SS = SSS(i, lightDir, atten);
                #else
                    half3 SS = IntegratePre(i, _Radius) * SSS(i, lightDir, atten);
                #endif
                //处理阴影
                //UNITY_LIGHT_ATTENUATION(atten,i,i.worldPos);
                
                fixed4 FinalColor = fixed4(diffuse + ambient + specular + fresnel + SS, 1.0);
                return FinalColor;
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
            sampler2D _MainTex;

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

            fixed4 frag(v2f i):SV_TARGET{
                fixed4 c = tex2D(_MainTex, i.tex);
                return fixed4(c.rgb * _OutlineColor.rgb, 1.0);
            }
            ENDCG
        }
    }
}