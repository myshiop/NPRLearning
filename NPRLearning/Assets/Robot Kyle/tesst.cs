using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;

public class tesst : MonoBehaviour
{
    // Start is called before the first frame update
    public Camera cam;
    public GameObject RenderTarget;
    int Count;
    string filePath = "";
    public int width;
    public int height;
    public int CaptureCount = 5;

    // Use this for initialization
    void Start()
    {
        Count = 0;
    }

    // Update is called once per frame
    void Update()
    {
        cam.gameObject.SetActive(true);
        filePath = string.Format("D:/TencentProject/NPRLearning/Assets/Resources/{0}.png", Count);
        if (Count < 5)
        {
            CamRender(cam, width, height, filePath);
            Debug.Log("GetCapture" + filePath);
            Count++;
        }
    }

    //渲染
    void CamRender(Camera cam, int width, int height, string filePath)
    {
        SceneView.lastActiveSceneView.camera.transform.Rotate(new Vector3(0, -180, 0));
        var bak_render_Tex = cam.targetTexture;
        var bak_cam_clearFlag = cam.clearFlags;
        var bak_RenderTexture_active = cam.activeTexture;

        var Tex = new Texture2D(width, height, TextureFormat.ARGB32, false);
        var render_Texture = RenderTexture.GetTemporary(width, height, 24, RenderTextureFormat.ARGB32);
        var grab_area = new Rect(0, 0, width, height);

        RenderTexture.active = render_Texture;
        cam.targetTexture = render_Texture;
        cam.clearFlags = CameraClearFlags.Nothing;

        //cam.backgroundColor = Color.white;
        cam.Render();

        Debug.Log("FinishRender");

        Tex.ReadPixels(grab_area, 0, 0);
        Tex.Apply();

        saveTexture2D(Tex, filePath);

        cam.targetTexture = bak_render_Tex;
        cam.clearFlags = bak_cam_clearFlag;
        RenderTexture.active = bak_RenderTexture_active;

        Texture2D.Destroy(Tex);
    }

    //保存图片
    public void saveTexture2D(Texture2D texture, string file)
    {
        byte[] bytes = texture.EncodeToPNG();
        UnityEngine.Object.Destroy(texture);
        System.IO.File.WriteAllBytes(file, bytes);
        Debug.Log("write to File over");
        UnityEditor.AssetDatabase.Refresh(); //自动刷新资源
    }

}
